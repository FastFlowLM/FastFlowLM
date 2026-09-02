/// \file granite_npu.cpp
/// \brief Host implementation of the Granite dense engine. See granite_npu.hpp.
///
/// Numerics follow the model the converter produced, not Granite's paper form.
/// q4nx-build folds Granite's four scalar multipliers into the weights, so what
/// the file holds is a Llama-scaled model and the emitted `config.json` says so
/// (`attention_multiplier` becomes the post-fold `head_dim ** -0.5`). Reading
/// the scale from the config therefore works for both a folded and an unfolded
/// build, which is why it is read rather than assumed.
///
/// q_proj / k_proj are stored in the PLAIN half-split arrangement, so nothing
/// has to be undone at load time. An earlier version of this file un-permuted
/// them on the belief that the converter had interleaved them, and that was the
/// bug that made the model ramble: the un-permutation introduced exactly the
/// scrambling it thought it was removing.
///
/// It was measured, not argued. Against a numpy forward pass that produces
/// correct text from these same bytes, layer 0 reads:
///
///     norm_in  cosine 1.00000000        (same input)
///     q        cosine -0.01162270       |q| 54.092 vs 54.091
///     k        cosine -0.08337054       |k| 302.411 vs 302.420
///     v        cosine 0.99999977
///
/// Identical norms with cosine ~0 is a permutation, not an arithmetic error,
/// and it hit exactly the two tensors that were being permuted while v, which
/// was not, matched to 1e-7.
///
/// The claim that the permutation had been "verified against
/// q4nx-build/tools/oracle_granite.py" was true and worthless: that oracle
/// reproduces the same broken output, because it makes the same assumption.

#include "models/granite/granite_npu.hpp"
#include "models/granite/q4nx_host.hpp"
#include "modules/gemm.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

/// \brief A persistent worker pool.
///
/// One decoder layer issues seven matvecs plus an attention pass, so a 40-layer
/// step is ~320 parallel regions. Creating and joining threads at each of those
/// costs more than the arithmetic inside many of them, so the threads are
/// created once and parked on a condition variable.
class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    size_t size() const { return workers_.size(); }

    /// \brief Run `body(begin, end)` over a partition of [0, n), and wait.
    /// \note Not reentrant: nested parallel_for would corrupt the shared state.
    ///       Nothing in this engine nests one.
    void run(size_t n, const std::function<void(size_t, size_t)>& body) {
        if (n == 0) return;
        const size_t parts = std::min(workers_.size() + 1, n);
        if (parts <= 1) {
            body(0, n);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            body_ = &body;
            n_ = n;
            parts_ = parts;
            chunk_ = (n + parts - 1) / parts;
            // Every worker wakes on every epoch and decrements exactly once,
            // whether or not it owns a part, so the count is the worker count
            // rather than the part count.
            remaining_.store(workers_.size(), std::memory_order_release);
            ++epoch_;
        }
        cv_.notify_all();

        run_part(0, n, (n + parts - 1) / parts, body);   // the caller takes part 0

        // Spin-then-yield: the parts are equal-sized, so the wait is short and
        // a condition variable here would cost more than it saves.
        while (remaining_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }

private:
    ThreadPool() {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        for (unsigned i = 0; i + 1 < hw; ++i)
            workers_.emplace_back([this, i] { worker(i + 1); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
            ++epoch_;
        }
        cv_.notify_all();
        for (auto& t : workers_) if (t.joinable()) t.join();
    }

    static void run_part(size_t part, size_t n, size_t chunk,
                         const std::function<void(size_t, size_t)>& body) {
        const size_t begin = part * chunk;
        if (begin >= n) return;
        body(begin, std::min(n, begin + chunk));
    }

    /// \brief Worker `id` owns part `id` of every epoch.
    ///
    /// Deliberately no work stealing. With stealing, a worker that has finished
    /// the last part of epoch N can still be looping on the shared counter when
    /// the caller returns and starts epoch N+1, reading `parts_`/`chunk_`/`body_`
    /// while they are being rewritten. Fixed parts make each worker read the
    /// epoch's state exactly once, under the mutex it was published with, and
    /// the parts are equal-sized so there is nothing to steal.
    void worker(size_t id) {
        size_t seen = 0;
        for (;;) {
            size_t n, chunk, parts;
            const std::function<void(size_t, size_t)>* body;
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [&] { return stop_ || epoch_ != seen; });
                if (stop_) return;
                seen = epoch_;
                n = n_; chunk = chunk_; parts = parts_; body = body_;
            }
            if (id < parts) run_part(id, n, chunk, *body);
            remaining_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_;
    const std::function<void(size_t, size_t)>* body_ = nullptr;
    size_t chunk_ = 0, n_ = 0, parts_ = 0, epoch_ = 0;
    std::atomic<size_t> next_{0};
    std::atomic<size_t> remaining_{0};
    bool stop_ = false;
};

/// \brief Split [0, n) across the pool.
template <typename F>
void parallel_for(size_t n, F&& body) {
    std::function<void(size_t, size_t)> fn(std::forward<F>(body));
    ThreadPool::instance().run(n, fn);
}

/// \brief Dot product of a bf16 row with a float vector.
///
/// bf16 -> float is just the top 16 bits of the float, so widening is a shift
/// rather than a conversion. Going through `static_cast<float>(bf16)` per
/// element costs a call into the bfloat16 type and dominates the inner loop;
/// doing it eight at a time with AVX2 is the single biggest host win available.
inline float dot_bf16(const bf16* w, const float* x, size_t n) {
#if defined(__AVX2__) || defined(_M_X64)
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(w);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t c = 0;
    for (; c + 16 <= n; c += 16) {
        // 8 bf16 -> 8 float: zero-extend to 32 bits, shift into the high half.
        __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + c));
        __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + c + 8));
        __m256 w0 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
        __m256 w1 = _mm256_castsi256_ps(
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + c), acc0);
        acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + c + 8), acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; c < n; ++c) {
        uint32_t bits = static_cast<uint32_t>(raw[c]) << 16;
        float wv;
        std::memcpy(&wv, &bits, sizeof(wv));
        sum += wv * x[c];
    }
    return sum;
#else
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(w);
    float sum = 0.0f;
    for (size_t c = 0; c < n; ++c) {
        uint32_t bits = static_cast<uint32_t>(raw[c]) << 16;
        float wv;
        std::memcpy(&wv, &bits, sizeof(wv));
        sum += wv * x[c];
    }
    return sum;
#endif
}

/// \brief y[0..rows) = W * x, with W row-major bf16 [rows][cols].
void matvec(const bf16* W, const float* x, float* y, size_t rows, size_t cols) {
    parallel_for(rows, [&](size_t begin, size_t end) {
        for (size_t r = begin; r < end; ++r) y[r] = dot_bf16(W + r * cols, x, cols);
    });
}

void rms_norm(const float* x, const float* weight, float* out, size_t n, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * x[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }


/// \brief Half-split RoPE, in place, over `heads` heads of `head_dim`.
void apply_rope(float* v, size_t heads, size_t head_dim, int pos, float theta) {
    const size_t half = head_dim / 2;
    for (size_t h = 0; h < heads; ++h) {
        float* p = v + h * head_dim;
        for (size_t i = 0; i < half; ++i) {
            const float freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
            const float angle = pos * freq;
            const float c = std::cos(angle), s = std::sin(angle);
            const float a = p[i], b = p[i + half];
            p[i] = a * c - b * s;
            p[i + half] = a * s + b * c;
        }
    }
}

}  // namespace

struct granite_npu::Impl {
    // geometry
    size_t hidden = 0, inter = 0, layers = 0, heads = 0, kv_heads = 0;
    size_t head_dim = 0, vocab = 0, q_dim = 0, kv_dim = 0, group = 1;
    float eps = 1e-5f, rope_theta = 10000.0f, attn_scale = 0.0f;
    size_t max_len = 4096, cur_len = 0, saved_len = 0;

    struct Layer {
        std::vector<bf16> wq, wk, wv, wo, wgate, wup, wdown;
        std::vector<float> ln_in, ln_post;
    };
    std::vector<Layer> layer;
    std::vector<bf16> embed, lm_head;
    std::vector<float> final_norm;

    // [layer][pos * kv_dim + i]
    std::vector<std::vector<bf16>> k_cache, v_cache;

    // scratch
    std::vector<float> x, h, q, k, v, attn, ff_gate, ff_up, logits;

    void allocate_scratch() {
        x.assign(hidden, 0.0f);
        h.assign(hidden, 0.0f);
        q.assign(q_dim, 0.0f);
        k.assign(kv_dim, 0.0f);
        v.assign(kv_dim, 0.0f);
        attn.assign(q_dim, 0.0f);
        ff_gate.assign(inter, 0.0f);
        ff_up.assign(inter, 0.0f);
        logits.assign(vocab, 0.0f);
    }

    void allocate_cache() {
        k_cache.assign(layers, {});
        v_cache.assign(layers, {});
        for (size_t l = 0; l < layers; ++l) {
            k_cache[l].assign(max_len * kv_dim, bf16(0.0f));
            v_cache[l].assign(max_len * kv_dim, bf16(0.0f));
        }
    }

    /// \brief One decoder step for `token` at position `cur_len`, filling logits.
    void step(int token) {
        const size_t pos = cur_len;
        for (size_t i = 0; i < hidden; ++i)
            x[i] = static_cast<float>(embed[static_cast<size_t>(token) * hidden + i]);

        for (size_t l = 0; l < layers; ++l) {
            Layer& L = layer[l];

            rms_norm(x.data(), L.ln_in.data(), h.data(), hidden, eps);
            matvec(L.wq.data(), h.data(), q.data(), q_dim, hidden);
            matvec(L.wk.data(), h.data(), k.data(), kv_dim, hidden);
            matvec(L.wv.data(), h.data(), v.data(), kv_dim, hidden);

            apply_rope(q.data(), heads, head_dim, static_cast<int>(pos), rope_theta);
            apply_rope(k.data(), kv_heads, head_dim, static_cast<int>(pos), rope_theta);

            for (size_t i = 0; i < kv_dim; ++i) {
                k_cache[l][pos * kv_dim + i] = static_cast<bf16>(k[i]);
                v_cache[l][pos * kv_dim + i] = static_cast<bf16>(v[i]);
            }

            // GQA: query head hh reads kv head hh / group
            parallel_for(heads, [&](size_t begin, size_t end) {
                std::vector<float> score(pos + 1);
                for (size_t hh = begin; hh < end; ++hh) {
                    const size_t kvh = hh / group;
                    const float* qh = q.data() + hh * head_dim;
                    float best = -INFINITY;
                    for (size_t t = 0; t <= pos; ++t) {
                        const bf16* kt = k_cache[l].data() + t * kv_dim + kvh * head_dim;
                        float dot = 0.0f;
                        for (size_t i = 0; i < head_dim; ++i)
                            dot += qh[i] * static_cast<float>(kt[i]);
                        score[t] = dot * attn_scale;
                        best = std::max(best, score[t]);
                    }
                    float denom = 0.0f;
                    for (size_t t = 0; t <= pos; ++t) {
                        score[t] = std::exp(score[t] - best);
                        denom += score[t];
                    }
                    float* out = attn.data() + hh * head_dim;
                    std::fill(out, out + head_dim, 0.0f);
                    for (size_t t = 0; t <= pos; ++t) {
                        const float wgt = score[t] / denom;
                        const bf16* vt = v_cache[l].data() + t * kv_dim + kvh * head_dim;
                        for (size_t i = 0; i < head_dim; ++i)
                            out[i] += wgt * static_cast<float>(vt[i]);
                    }
                }
            });

            matvec(L.wo.data(), attn.data(), h.data(), hidden, q_dim);
            for (size_t i = 0; i < hidden; ++i) x[i] += h[i];

            rms_norm(x.data(), L.ln_post.data(), h.data(), hidden, eps);
            matvec(L.wgate.data(), h.data(), ff_gate.data(), inter, hidden);
            matvec(L.wup.data(), h.data(), ff_up.data(), inter, hidden);
            for (size_t i = 0; i < inter; ++i) ff_gate[i] = silu(ff_gate[i]) * ff_up[i];
            matvec(L.wdown.data(), ff_gate.data(), h.data(), hidden, inter);
            for (size_t i = 0; i < hidden; ++i) x[i] += h[i];
        }

        rms_norm(x.data(), final_norm.data(), h.data(), hidden, eps);
        matvec(lm_head.data(), h.data(), logits.data(), vocab, hidden);
        cur_len = pos + 1;
    }
};

granite_npu::granite_npu(LM_Config config, npu_xclbin_manager* npu_instance, int MAX_L)
    : _impl(new Impl()) {
    Impl& I = *_impl;
    I.hidden = config.get("hidden_size");
    I.inter = config.get("intermediate_size");
    I.layers = config.get("num_hidden_layers");
    I.heads = config.get("num_attention_heads");
    I.kv_heads = config.get("num_key_value_heads", static_cast<u32>(I.heads));
    I.vocab = config.get("vocab_size");
    I.head_dim = config.get("head_dim", static_cast<u32>(I.hidden / (I.heads ? I.heads : 1)));
    I.eps = config.get<f32>("rms_norm_eps", 1e-5f);
    I.rope_theta = config.get<f32>("rope_theta", 10000.0f);
    I.q_dim = I.heads * I.head_dim;
    I.kv_dim = I.kv_heads * I.head_dim;
    I.group = I.kv_heads ? I.heads / I.kv_heads : 1;
    // Post-fold: the converter writes head_dim**-0.5 here for a folded build.
    I.attn_scale = config.get<f32>("attention_multiplier",
                                   1.0f / std::sqrt(static_cast<float>(I.head_dim)));
    I.max_len = static_cast<size_t>(MAX_L > 0 ? MAX_L : 4096);

    header_print("FLM", "granite (host engine): hidden " << I.hidden << ", layers " << I.layers
                        << ", heads " << I.heads << "/" << I.kv_heads
                        << ", head_dim " << I.head_dim << ", attn_scale " << I.attn_scale);

    I.allocate_scratch();
    I.allocate_cache();
}

granite_npu::~granite_npu() { delete _impl; }

void granite_npu::load_weights(Q4NX& q4nx) {
    Impl& I = *_impl;

    auto read = [&](const std::string& name, size_t rows, size_t cols,
                    std::vector<bf16>& out, bool undo_rope_permutation) {
        bytes raw;
        q4nx.load_weights(raw, name);
        out.assign(rows * cols, bf16(0.0f));
        if (q4nx_host::is_tiled(raw.size(), rows, cols)) {
            q4nx_host::dequantize(raw.data(), raw.size(), rows, cols, out.data());
        } else if (raw.size() == rows * cols * sizeof(bf16)) {
            std::memcpy(out.data(), raw.data(), raw.size());   // bf16 passthrough
        } else {
            throw std::runtime_error("granite: unexpected size for " + name);
        }
        if (undo_rope_permutation) {
            // The converter stores q/k as '(g p q) c -> (g q p) c' with
            // p = head_dim/2, q = 2. Undo it so the forward pass can use the
            // plain half-split rotation.
            std::vector<bf16> tmp(out.size());
            const size_t hd = I.head_dim, half = hd / 2;
            const size_t n_heads = rows / hd;
            for (size_t hh = 0; hh < n_heads; ++hh)
                for (size_t p = 0; p < half; ++p)
                    for (size_t qq = 0; qq < 2; ++qq) {
                        const size_t src = hh * hd + qq * half + p;
                        const size_t dst = hh * hd + 2 * p + qq;
                        std::memcpy(tmp.data() + dst * cols, out.data() + src * cols,
                                    cols * sizeof(bf16));
                    }
            out.swap(tmp);
        }
    };

    auto read_norm = [&](const std::string& name, size_t n, std::vector<float>& out) {
        bytes raw;
        q4nx.load_weights(raw, name);
        out.assign(n, 0.0f);
        const bf16* src = reinterpret_cast<const bf16*>(raw.data());
        const size_t have = std::min(n, raw.size() / sizeof(bf16));
        for (size_t i = 0; i < have; ++i) out[i] = static_cast<float>(src[i]);
    };

    header_print("FLM", "granite: dequantizing weights to bf16 (host)...");
    read("model.embed_tokens.weight", I.vocab, I.hidden, I.embed, false);
    read("lm_head.weight", I.vocab, I.hidden, I.lm_head, false);
    read_norm("model.norm.weight", I.hidden, I.final_norm);

    I.layer.resize(I.layers);
    for (size_t l = 0; l < I.layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        Impl::Layer& L = I.layer[l];
        read(p + "self_attn.q_proj.weight", I.q_dim, I.hidden, L.wq, false);
        read(p + "self_attn.k_proj.weight", I.kv_dim, I.hidden, L.wk, false);
        read(p + "self_attn.v_proj.weight", I.kv_dim, I.hidden, L.wv, false);
        read(p + "self_attn.o_proj.weight", I.hidden, I.q_dim, L.wo, false);
        read(p + "mlp.gate_proj.weight", I.inter, I.hidden, L.wgate, false);
        read(p + "mlp.up_proj.weight", I.inter, I.hidden, L.wup, false);
        read(p + "mlp.down_proj.weight", I.hidden, I.inter, L.wdown, false);
        read_norm(p + "input_layernorm.weight", I.hidden, L.ln_in);
        read_norm(p + "post_attention_layernorm.weight", I.hidden, L.ln_post);
    }
    header_print("FLM", "granite: weights ready (" << I.layers << " layers)");
}

namespace {
/// Every id the engine has seen this sequence: the prefill prompt followed by
/// each sampled token. A decode-step dump has to carry the whole history, not
/// just the new token, because the oracle replays it from scratch.
std::vector<int> g_dump_ids;

/// Writes (start_pos, ids[], logits[]) for the oracle to replay. `step` is 0 for
/// prefill and 1.. for decode steps; GRANITE_DUMP_STEP selects which one to
/// capture, so a single run can be inspected at any point in the generation.
void granite_dump(const std::vector<int>& ids, const std::vector<float>& logits,
                  size_t vocab, uint32_t start_pos, int step) {
    const char* path = std::getenv("GRANITE_DEBUG_DUMP");
    if (path == nullptr) return;
    const char* want = std::getenv("GRANITE_DUMP_STEP");
    if (step != (want ? std::atoi(want) : 0)) return;
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        header_print("WARNING", "granite: cannot open GRANITE_DEBUG_DUMP path " << path);
        return;
    }
    const uint32_t n_ids = static_cast<uint32_t>(ids.size());
    const uint32_t n_log = static_cast<uint32_t>(vocab);
    f.write(reinterpret_cast<const char*>(&start_pos), 4);
    f.write(reinterpret_cast<const char*>(&n_ids), 4);
    f.write(reinterpret_cast<const char*>(&n_log), 4);
    f.write(reinterpret_cast<const char*>(ids.data()), n_ids * sizeof(int));
    f.write(reinterpret_cast<const char*>(logits.data()), n_log * sizeof(float));
    header_print("FLM", "granite: dumped step " << step << " -- " << n_ids
                        << " ids + " << n_log << " logits to " << path);
}
}  // namespace

buffer<bf16> granite_npu::forward(int ids) {
    Impl& I = *_impl;
    if (I.cur_len >= I.max_len) I.cur_len = I.max_len - 1;
    I.step(ids);
    // The token just consumed becomes part of the history the oracle replays.
    g_dump_ids.push_back(ids);
    static int decode_step = 0;
    granite_dump(g_dump_ids, I.logits, I.vocab,
                 static_cast<uint32_t>(g_dump_ids.size() - 1), ++decode_step);
    buffer<bf16> out(I.vocab);
    for (size_t i = 0; i < I.vocab; ++i) out[i] = static_cast<bf16>(I.logits[i]);
    return out;
}

buffer<bf16> granite_npu::prefill(std::vector<int>& ids, void* /*payload*/) {
    Impl& I = *_impl;
    const size_t start = I.cur_len;
    for (size_t t = 0; t < ids.size(); ++t) {
        if (I.cur_len >= I.max_len) break;
        I.step(ids[t]);
    }
    buffer<bf16> out(I.vocab);
    for (size_t i = 0; i < I.vocab; ++i) out[i] = static_cast<bf16>(I.logits[i]);

    // The prompt the engine actually saw, and the logits it produced from it,
    // so they can be diffed against the Python oracle rather than argued about
    // from output text. GRANITE_DUMP_STEP selects prefill (0) or a decode step.
    g_dump_ids = ids;
    granite_dump(ids, I.logits, I.vocab, static_cast<uint32_t>(start), 0);
    return out;
}

void granite_npu::set_context_length(int L) {
    _impl->cur_len = static_cast<size_t>(std::max(0, L));
}

void granite_npu::clear_context() { _impl->cur_len = 0; }

int granite_npu::get_current_context_length() { return static_cast<int>(_impl->cur_len); }

void granite_npu::update_max_length(uint32_t MAX_L) {
    Impl& I = *_impl;
    if (MAX_L == I.max_len) return;
    I.max_len = MAX_L;
    I.allocate_cache();
    if (I.cur_len > I.max_len) I.cur_len = I.max_len;
}

buffer<bf16> granite_npu::get_k_cache(int layer_idx, int idx) {
    Impl& I = *_impl;
    return buffer<bf16>(I.k_cache[layer_idx].data() + static_cast<size_t>(idx) * I.kv_dim,
                        I.kv_dim);
}

buffer<bf16> granite_npu::get_v_cache(int layer_idx, int idx) {
    Impl& I = *_impl;
    return buffer<bf16>(I.v_cache[layer_idx].data() + static_cast<size_t>(idx) * I.kv_dim,
                        I.kv_dim);
}

int granite_npu::checkpoint() {
    _impl->saved_len = _impl->cur_len;
    return static_cast<int>(_impl->saved_len);
}

int granite_npu::restore() {
    _impl->cur_len = _impl->saved_len;
    return static_cast<int>(_impl->cur_len);
}
