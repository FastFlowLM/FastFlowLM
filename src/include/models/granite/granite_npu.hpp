/// \file granite_npu.hpp
/// \brief IBM Granite (dense) engine.
///
/// Unlike the shipped engines this one is **open and host-side**. It exists
/// because granite-4.2-3B cannot run on any compiled design FastFlowLM ships:
/// it needs head_dim 64 at hidden 2560, and every shipped head_dim-64 design is
/// hidden 2048 while every design at hidden >= 2560 is head_dim 128. Those sets
/// are disjoint, hidden can only be padded upward, and head_dim is intrinsic to
/// RoPE and cannot be padded at all.
///
/// `llama_npu.dll` additionally whitelists `hidden_size` to {2048, 3072, 4096}
/// and refuses 2560 outright. That gate lives inside the closed engine; this
/// one has no such restriction and runs the model at its native geometry.
///
/// Correctness first, speed second: this is milestone 1 of a staged plan, and
/// its job is to make `flm.exe` produce the right tokens for granite so that
/// every later stage that moves work onto the array has a reference to diff
/// against **inside FLM's own process**. Nothing here touches the NPU yet.
#pragma once

#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

/// \brief Granite dense engine (host implementation of causal_lm).
class granite_npu : public causal_lm {
public:
    /// \param config       the model configuration
    /// \param npu_instance accepted for interface parity with the shipped
    ///                     engines; unused, this milestone runs on the host
    /// \param MAX_L        maximum context length
    granite_npu(LM_Config config, npu_xclbin_manager* npu_instance, int MAX_L = 4096);
    ~granite_npu();

    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    void set_context_length(int L) override;
    void load_weights(Q4NX& q4nx) override;
    void clear_context() override;

    buffer<bf16> get_k_cache(int layer_idx, int idx) override;
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    void update_max_length(uint32_t MAX_L) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

private:
    struct Impl;
    Impl* _impl;
};
