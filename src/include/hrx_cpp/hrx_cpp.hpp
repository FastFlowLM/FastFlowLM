/// \file hrx_cpp.hpp
/// \brief Minimal C++ `namespace hrx` providing the device/buffer/kernel/run
///        API that FastFlowLM uses, implemented directly on top of libhrx
///        (hrx_runtime.h).
///
/// NPU control code goes straight from npu_sequence::dump() into an HRX XADX
/// "direct executable"; there is no separate assembler step.
///
/// Coherence model: this amdxdna device is a single HOST_ONLY heap (cached
/// host DRAM the NPU can snoop). Buffers are allocated HOST_VISIBLE |
/// HOST_CACHED | DEVICE_VISIBLE (0x1A), mapped once (persistent). The heap
/// is not HOST_COHERENT, so FastFlowLM's sync_to_device()/sync_from_device()
/// map to hrx_buffer_flush_range()/hrx_buffer_invalidate_range(). Dispatch is
/// hrx_stream_dispatch() + hrx_stream_flush()/hrx_stream_wait().
#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "hrx_amdxdna.h"
#include "hrx_runtime.h"

// ---- ert_cmd_state: command states that FLM's npu_utils returns/maps.
#ifndef FLM_ERT_CMD_STATE_DEFINED
#define FLM_ERT_CMD_STATE_DEFINED
enum ert_cmd_state {
    ERT_CMD_STATE_NEW = 1,
    ERT_CMD_STATE_QUEUED = 2,
    ERT_CMD_STATE_RUNNING = 3,
    ERT_CMD_STATE_COMPLETED = 4,
    ERT_CMD_STATE_ERROR = 5,
    ERT_CMD_STATE_ABORT = 6,
    ERT_CMD_STATE_SUBMITTED = 7,
    ERT_CMD_STATE_TIMEOUT = 8,
    ERT_CMD_STATE_NORESPONSE = 9,
    ERT_CMD_STATE_SKERROR = 10,
    ERT_CMD_STATE_SKCRASHED = 11,
    ERT_CMD_STATE_MAX = 12,
};
#endif

namespace hrx {

// ---------------------------------------------------------------------------
// Process-wide HRX runtime (one device + one stream), lazily initialized.
// ---------------------------------------------------------------------------
class Runtime {
public:
    static Runtime& get() {
        static Runtime r;
        return r;
    }
    hrx_device_t dev = nullptr;
    hrx_stream_t stream = nullptr;
    bool ok = false;

    void ensure() {
        if (dev) return;
        hrx_status_t init_status = hrx_gpu_initialize(0);
        const bool initialized =
            hrx_status_is_ok(init_status) ||
            hrx_status_code(init_status) == HRX_STATUS_ALREADY_EXISTS;
        hrx_status_ignore(init_status);
        if (initialized &&
            hrx_status_is_ok(hrx_gpu_device_get(0, &dev)) &&
            hrx_status_is_ok(hrx_stream_create(dev, 0, &stream))) {
            ok = true;
            if (std::getenv("HRX_DEBUG"))
                std::fprintf(stderr, "[hrx] device+stream initialized (Runtime@%p)\n",
                             (void*)this);
        } else {
            std::fprintf(stderr, "[hrx] device init FAILED\n");
            ok = false;
        }
    }

private:
    Runtime() = default;
};

inline Runtime& rt() {
    Runtime& r = Runtime::get();
    r.ensure();
    return r;
}

// Report (do not swallow) an HRX error. Returns true if status was an error.
// FLM dispatch silently ignored synchronize/dispatch failures, which turns a
// failed ERT_CMD_CHAIN (e.g. a missing host patch table) into silent no-op
// dispatches -> garbage output at full speed. Always surface these.
inline bool hrx_report(hrx_status_t s, const char* where) {
    if (hrx_status_is_ok(s)) return false;
    char* m = nullptr;
    size_t mn = 0;
    hrx_status_to_string(s, &m, &mn);
    std::fprintf(stderr, "[hrx][ERROR] %s: %s\n", where, m ? m : "?");
    hrx_status_free_message(m);
    hrx_status_ignore(s);
    return true;
}

// ---------------------------------------------------------------------------
// Executable cache: build one HRX XADX executable per distinct executable
// identity (xclbin + control program + host patch table) and resolve its export
// ordinal once.
// ---------------------------------------------------------------------------
struct CachedExe {
    hrx_executable_t exe = nullptr;
    uint32_t ord = 0;
};

inline void append_key_bytes(std::string& key, const void* data, size_t byte_count) {
    const uint64_t length = static_cast<uint64_t>(byte_count);
    key.append(reinterpret_cast<const char*>(&length), sizeof(length));
    if (data && byte_count) {
        key.append(reinterpret_cast<const char*>(data), byte_count);
    }
}

inline hrx_executable_t build_or_get_executable(
    const std::vector<uint8_t>& xclbin_bytes, const uint32_t* cc, size_t n,
    uint32_t* ord_out) {
    static std::mutex mu;
    static std::unordered_map<std::string, CachedExe> cache;
    std::string key;
    key.reserve(xclbin_bytes.size() + n * sizeof(uint32_t) +
                2 * sizeof(uint64_t));
    append_key_bytes(key, xclbin_bytes.data(), xclbin_bytes.size());
    append_key_bytes(key, cc, n * sizeof(uint32_t));
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (ord_out) *ord_out = it->second.ord;
        return it->second.exe;
    }
    hrx_const_byte_span_t xclbin = {xclbin_bytes.data(), xclbin_bytes.size()};
    hrx_amdxdna_executable_run_t run = {};
    run.record_length = sizeof(run);
    run.abi_version = HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0;
    run.transaction = {reinterpret_cast<const uint8_t*>(cc),
                       n * sizeof(uint32_t)};
    hrx_amdxdna_executable_entry_point_t entry_point = {};
    entry_point.record_length = sizeof(entry_point);
    entry_point.abi_version = HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0;
    entry_point.name = {"MLIR_AIE", std::strlen("MLIR_AIE")};
    entry_point.context_mode = HRX_AMDXDNA_CONTEXT_MODE_CREATE;
    entry_point.runs = &run;
    entry_point.run_count = 1;
    hrx_amdxdna_executable_create_params_t params = {};
    params.record_length = sizeof(params);
    params.abi_version = HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0;
    params.xclbins = &xclbin;
    params.xclbin_count = 1;
    params.entry_points = &entry_point;
    params.entry_point_count = 1;
    hrx_executable_t exe = nullptr;
    uint32_t ord = 0;
    hrx_status_t create_status = hrx_amdxdna_executable_create(
        rt().dev, &params, &exe);
    if (hrx_report(create_status, "hrx_amdxdna_executable_create")) {
        exe = nullptr;
    }
    if (exe && hrx_report(hrx_executable_lookup_export_by_name(
                              exe, "MLIR_AIE", &ord),
                          "hrx_executable_lookup_export_by_name")) {
        hrx_executable_release(exe);
        exe = nullptr;
    }
    cache.emplace(std::move(key), CachedExe{exe, ord});
    if (ord_out) *ord_out = ord;
    return exe;
}

// ---------------------------------------------------------------------------
// uuid / xclbin / device / hw_context
// ---------------------------------------------------------------------------
class uuid {
public:
    unsigned char m_uuid[16] = {0};
};

class xclbin {
public:
    std::shared_ptr<std::vector<uint8_t>> bytes_ =
        std::make_shared<std::vector<uint8_t>>();

    xclbin() = default;
    explicit xclbin(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("hrx::xclbin: cannot open " + path);
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n > 0) {
            bytes_->resize(static_cast<size_t>(n));
            size_t rd = std::fread(bytes_->data(), 1, bytes_->size(), f);
            (void)rd;
        }
        std::fclose(f);
    }

    // FLM searches kernels for one whose name starts with "MLIR_AIE"; the HRX
    // dispatch path always uses the "MLIR_AIE" export, so a single placeholder
    // kernel is sufficient (matches the proven interposer behavior).
    class kernel {
    public:
        std::string name = "MLIR_AIE";
        std::string get_name() const { return name; }
    };
    std::vector<kernel> get_kernels() const { return {kernel{}}; }
    uuid get_uuid() const { return uuid{}; }
    const std::vector<uint8_t>& bytes() const { return *bytes_; }
    std::shared_ptr<std::vector<uint8_t>> bytes_shared() const { return bytes_; }
};

namespace info {
// Argument to device::get_info<>(). FLM only queries the human-readable
// device name for diagnostics.
enum class device { name, architecture };
}

class device {
public:
    device() = default;
    explicit device(unsigned int /*index*/) { rt(); }
    uuid register_xclbin(const xclbin& /*xc*/) { return uuid{}; }
    void reset() {}

    // Returns a human-readable device identity string for diagnostics: the
    // IREE HAL device name (e.g. "amdxdna") via hrx_device_get_property().
    template <info::device P>
    std::string get_info() const {
        char buf[128] = {0};
        hrx_device_get_property(
            rt().dev,
            P == info::device::architecture ? HRX_DEVICE_PROPERTY_ARCHITECTURE
                                            : HRX_DEVICE_PROPERTY_NAME,
            buf, sizeof(buf));
        return std::string(buf);
    }
};

class hw_context {
public:
    std::shared_ptr<std::vector<uint8_t>> xclbin_bytes_;

    hw_context() = default;
    hw_context(const device& /*dev*/, const xclbin& xc)
        : xclbin_bytes_(xc.bytes_shared()) {}
    // The (device, uuid) form is accepted for source compatibility; it carries
    // no xclbin bytes, so prefer the (device, xclbin) form.
    hw_context(const device& /*dev*/, const uuid& /*id*/) {}

    const std::vector<uint8_t>& xclbin_bytes() const {
        static const std::vector<uint8_t> empty;
        return xclbin_bytes_ ? *xclbin_bytes_ : empty;
    }
};

// ---------------------------------------------------------------------------
// Per-buffer host<->device coherence state (dirty tracking).
//
// This path uses ONE persistent host-mapped, device-visible buffer per BO;
// coherence is kept purely with clflush-style range ops. The correctness
// comes entirely from GATING those ops per buffer:
//
//   host_dirty : the host wrote bytes that are not yet on the device. Must be
//                flushed (h2d) before any dispatch reads the buffer, and only
//                then.  Cleared by the flush.
//   dev_dirty  : a dispatch wrote bytes that are not yet visible to the host.
//                Must be invalidated (d2h) before the host reads, and only
//                then.  Cleared by the invalidate.
//
// Why gating matters (this is the multi-turn bug):
//   * Unconditional flush-before-dispatch pushes a STALE host copy over a
//     device-resident buffer (e.g. kv_caches the previous layer just wrote) ->
//     corrupts the KV cache.
//   * Unconditional invalidate-on-read DROPS the host's own not-yet-flushed
//     writes -> the device never sees them on the next turn.
// This path avoids both by only acting when the matching dirty bit is set; on a
// single turn the bits happen to line up either way, but turn>1 reuses buffers
// whose bits diverge, which is why only later turns broke.
struct BufCoh {
    bool host_dirty = true;   // freshly-allocated content must reach the device once
    bool dev_dirty = false;
};
inline std::mutex& buf_coh_mu() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<hrx_buffer_t, BufCoh>& buf_coh() {
    static std::unordered_map<hrx_buffer_t, BufCoh> m;
    return m;
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
class bo {
public:
    hrx_buffer_t hbuf_ = nullptr;
    void* mapped_ = nullptr;
    size_t size_ = 0;
    bool owns_ = false;

    bo() = default;
    virtual ~bo() {
        if (owns_ && hbuf_) {
            {
                std::lock_guard<std::mutex> lk(buf_coh_mu());
                buf_coh().erase(hbuf_);
            }
            hrx_buffer_release(hbuf_);
        }
        hbuf_ = nullptr;
        mapped_ = nullptr;
    }
    bo(const bo&) = delete;
    bo& operator=(const bo&) = delete;

    template <typename T>
    T map() {
        return reinterpret_cast<T>(mapped_);
    }
    size_t size() const { return size_; }
    hrx_buffer_t handle() const { return hbuf_; }

    // sync_to_device(): FLM's buffer abstraction uses whole-BO syncs, matching
    // xrt::bo::sync(XCL_BO_SYNC_BO_TO_DEVICE). Record that the host has new
    // contents and defer the actual h2d flush to the dispatch that consumes the
    // buffer.
    void flush() {
        if (!hbuf_) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        st.host_dirty = true;
        st.dev_dirty = false;  // host is now the source of truth
    }
    // sync_from_device(): the host wants to read. Only invalidate if a dispatch
    // actually produced new device data (dev_dirty); otherwise this would drop
    // the host's own writes.
    void invalidate() {
        if (!hbuf_) return;
        std::lock_guard<std::mutex> lk(buf_coh_mu());
        BufCoh& st = buf_coh()[hbuf_];
        if (st.dev_dirty) {
            hrx_status_t s = hrx_buffer_invalidate_range(hbuf_, 0, size_);
            if (hrx_report(s, "bo::invalidate hrx_buffer_invalidate_range")) {
                return;
            }
            st.dev_dirty = false;
        }
    }
};

namespace ext {
class bo : public hrx::bo {
public:
    bo(const device& /*dev*/, size_t sz) {
        Runtime& r = rt();
        size_ = sz;
        owns_ = true;
        if (!r.ok) throw std::runtime_error("hrx::ext::bo: HRX device unavailable");
        // HOST_VISIBLE | HOST_CACHED | DEVICE_VISIBLE (0x1A). This device has
        // one HOST_ONLY heap; HOST_LOCAL / HOST_COHERENT / DEVICE_LOCAL are
        // rejected. Coherence is flush after host writes and invalidate after
        // device writes. Persistent map via hrx_buffer_map_with_mode.
        hrx_status_t s = hrx_buffer_allocate(
            r.stream, sz,
            HRX_MEMORY_TYPE_HOST_VISIBLE |
                HRX_MEMORY_TYPE_HOST_CACHED |
                HRX_MEMORY_TYPE_DEVICE_VISIBLE,
            HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
            &hbuf_);
        if (!hrx_status_is_ok(s) || !hbuf_) {
            hrx_status_ignore(s);
            throw std::runtime_error("hrx::ext::bo: hrx_buffer_allocate failed");
        }
        void* p = nullptr;
        s = hrx_buffer_map_with_mode(hbuf_, HRX_MAPPING_MODE_PERSISTENT,
                                     HRX_MAP_READ | HRX_MAP_WRITE, 0, sz, &p);
        if (!hrx_status_is_ok(s) || !p) {
            hrx_status_ignore(s);
            hrx_buffer_release(hbuf_);
            hbuf_ = nullptr;
            throw std::runtime_error("hrx::ext::bo: map_persistent failed");
        }
        mapped_ = p;
        std::memset(p, 0, sz);
        // The zeroed contents are host-side only until the first dispatch; mark
        // host_dirty so the gated h2d flushes them to the device exactly once.
        {
            std::lock_guard<std::mutex> lk(buf_coh_mu());
            buf_coh()[hbuf_] = BufCoh{/*host_dirty=*/true, /*dev_dirty=*/false};
        }
    }
};
}  // namespace ext

// ---------------------------------------------------------------------------
// run / runlist
// ---------------------------------------------------------------------------
// Gated host->device flush for a dispatch's bindings: flush only the buffers the
// host has dirtied (host_dirty), exactly once, right before the device reads
// them. Device-owned buffers (host_dirty == false) are left untouched so they
// are never clobbered. Mirrors the shim's "h2d only dirty inputs" phase.
inline bool hrx_h2d_bindings(const std::vector<hrx_buffer_ref_t>& binds) {
    bool err = false;
    std::lock_guard<std::mutex> lk(buf_coh_mu());
    auto& coh = buf_coh();
    for (const auto& b : binds) {
        if (!b.buffer) continue;
        BufCoh& st = coh[b.buffer];
        if (st.host_dirty) {
            hrx_status_t s =
                hrx_buffer_flush_range(b.buffer, b.offset, b.length);
            if (hrx_report(s, "run::record hrx_buffer_flush_range")) {
                err = true;
                continue;
            }
            st.host_dirty = false;
        }
    }
    return err;
}

// After a dispatch completes, the device holds the freshest copy of every bound
// buffer: clear host_dirty and mark dev_dirty so the next host read invalidates
// (lazily, via sync_from_device). Mirrors the shim's post-dispatch readback
// bookkeeping. We over-approximate by treating every binding as a potential
// output; that is safe because an extra dev_dirty only triggers a redundant
// invalidate that re-reads identical bytes.
inline void hrx_mark_dispatched(const std::vector<hrx_buffer_ref_t>& binds) {
    std::lock_guard<std::mutex> lk(buf_coh_mu());
    auto& coh = buf_coh();
    for (const auto& b : binds) {
        if (!b.buffer) continue;
        BufCoh& st = coh[b.buffer];
        st.host_dirty = false;
        st.dev_dirty = true;
    }
}

class run {
public:
    hrx_executable_t exe_ = nullptr;
    uint32_t ord_ = 0;
    std::vector<hrx_buffer_ref_t> binds_;
    bool submitted_ = false;
    bool flushed_ = false;
    bool submit_error_ = false;

    run() = default;
    explicit run(hrx_executable_t exe, uint32_t ord) : exe_(exe), ord_(ord) {}

    void add_binding(hrx_buffer_t b, size_t size) {
        binds_.push_back({b, 0, size});
    }

    bool record() {
        submitted_ = false;
        flushed_ = false;
        submit_error_ = false;
        if (!exe_) {
            std::fprintf(stderr, "[hrx][ERROR] run::record with null executable\n");
            submit_error_ = true;
            return false;
        }
        // Zero bindings is valid: RTP/control-only runs such as
        // set_layer_rtp.create_run() and gemma4e layer_pre_load.create_run().
        // Opcode scalars live in the TXN stream, not as BO bindings.
        if (hrx_h2d_bindings(binds_)) {
            submit_error_ = true;
            return false;
        }
        hrx_dispatch_config_t cfg = {{1, 1, 1}, {1, 1, 1}, 0};
        hrx_status_t s = hrx_stream_dispatch(rt().stream, exe_, ord_, &cfg,
                                             nullptr, 0, binds_.data(),
                                             binds_.size(), HRX_DISPATCH_FLAG_NONE);
        if (hrx_report(s, "run::record hrx_stream_dispatch")) {
            submit_error_ = true;
            return false;
        }
        submitted_ = true;
        return true;
    }

    void start() {
        if (!record()) return;
        hrx_status_t s = hrx_stream_flush(rt().stream);
        if (hrx_report(s, "run::start hrx_stream_flush")) {
            submit_error_ = true;
            return;
        }
        flushed_ = true;
    }

    ert_cmd_state wait() {
        if (submit_error_) return ERT_CMD_STATE_ERROR;
        if (!flushed_) {
            return submitted_ ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
        }
        hrx_status_t s = hrx_stream_wait(rt().stream);
        bool err = hrx_report(s, "run::wait hrx_stream_wait");
        if (!err && submitted_) hrx_mark_dispatched(binds_);
        return err ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
    }
};

class runlist {
public:
    std::vector<run> runs_;
    bool flushed_ = false;
    bool submit_error_ = false;

    runlist() = default;
    explicit runlist(const hw_context& /*ctx*/) {}

    void add(const run& r) { runs_.push_back(r); }
    void add(run&& r) { runs_.push_back(std::move(r)); }
    void reset() {
        runs_.clear();
        flushed_ = false;
        submit_error_ = false;
    }

    void execute() {
        flushed_ = false;
        submit_error_ = false;
        bool any_submitted = false;
        for (auto& r : runs_) {
            if (r.record()) {
                any_submitted = true;
            } else {
                submit_error_ = true;
            }
        }
        if (!any_submitted) return;
        hrx_status_t s = hrx_stream_flush(rt().stream);
        if (hrx_report(s, "runlist::execute hrx_stream_flush")) {
            submit_error_ = true;
            return;
        }
        flushed_ = true;
    }
    ert_cmd_state wait() {
        if (!flushed_) {
            return submit_error_ ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
        }
        hrx_status_t s = hrx_stream_wait(rt().stream);
        bool err = hrx_report(s, "runlist::wait hrx_stream_wait");
        if (!err) {
            for (auto& r : runs_) {
                if (r.submitted_) hrx_mark_dispatched(r.binds_);
            }
        }
        return (submit_error_ || err) ? ERT_CMD_STATE_ERROR : ERT_CMD_STATE_COMPLETED;
    }
};

}  // namespace hrx
