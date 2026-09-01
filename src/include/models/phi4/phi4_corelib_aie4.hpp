#pragma once

#include <causal_lm.hpp>
#include <corelib/corelib_runtime.hpp>
#include <lm_config.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace flm::phi4 {

struct Phi4DebugSnapshot {
    std::int64_t live_rows;
    std::int64_t position;
    std::vector<std::uint16_t> layer0_k;
    std::vector<std::uint16_t> layer0_v;
    std::vector<std::uint16_t> layer31_k;
    std::vector<std::uint16_t> layer31_v;
    std::vector<std::uint16_t> last_hidden;
    std::vector<std::uint16_t> logits;
};

struct Phi4Aie4Metrics {
    std::uint64_t model_load_ns = 0;
    std::uint64_t weight_pack_ns = 0;
    std::uint64_t packed_weight_bytes = 0;
    std::uint64_t mapped_source_bytes = 0;
    std::uint64_t kv_bytes = 0;
    std::uint64_t scratch_bytes = 0;
    std::uint64_t device_tensor_create_count = 0;
    std::uint64_t weight_create_count = 0;
    std::uint64_t dispatch_count = 0;
    std::uint64_t synchronize_count = 0;
    std::uint64_t v_read_calls = 0;
    std::uint64_t v_write_calls = 0;
    std::uint64_t v_bytes = 0;
    std::uint64_t v_scatter_ns = 0;
};

class phi4_corelib_aie4 final : public causal_lm {
public:
    phi4_corelib_aie4(
        LM_Config config,
        std::filesystem::path model_path,
        std::shared_ptr<corelib::CorelibRuntime> runtime,
        std::uint32_t max_length = 4096);
    ~phi4_corelib_aie4() override;

    buffer<bf16> forward(int id) override;
    buffer<bf16> prefill(
        std::vector<int>& ids,
        void* payload = nullptr) override;
    void set_context_length(int length) override;
    void load_weights(Q4NX& q4nx) override;
    void update_max_length(std::uint32_t max_length) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int layer, int index) override;
    buffer<bf16> get_v_cache(int layer, int index) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;
    const Phi4Aie4Metrics& metrics() const noexcept;
#ifdef DEV_BUILD
    Phi4DebugSnapshot debug_snapshot() const;
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#if defined(FLM_CORELIB_TESTING)
namespace testing {

[[noreturn]] void ApplyCorelibFailurePolicyForTest(
    const std::shared_ptr<corelib::CorelibRuntime>& runtime,
    const corelib::CorelibError& error,
    bool synchronize_in_progress,
    const corelib::StepSubmissionState& submission,
    std::string phase,
    std::optional<int> layer,
    std::int64_t rows,
    std::int64_t position);

}  // namespace testing
#endif

}  // namespace flm::phi4
