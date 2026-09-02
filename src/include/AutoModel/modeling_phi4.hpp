/// \file phi4.hpp
/// \brief phi4 class
/// \author FastFlowLM Team
/// \date 2025-09-04
/// \version 0.9.25
/// \note This is a source file for the phi4 class
#pragma once
#include "AutoModel/automodel.hpp"

#if defined(FLM_ENABLE_CORELIB_AIE4)
#include "corelib/corelib_runtime.hpp"
#include "models/phi4/phi4_corelib_aie4.hpp"
#include "models/phi4/phi4_corelib_aie4_tuning.hpp"
#include <optional>
#endif

#if defined(FLM_CORELIB_TESTING)
#include <filesystem>
#include <functional>
#endif

#if defined(FLM_CORELIB_TESTING)
namespace flm::phi4::testing {
class Phi4FrontendTestAccess;
}
#endif

/************              phi4 family            **************/
class Phi4 : public AutoModel {
private:
    void setup_tokenizer(
        const std::string& model_path,
        bool require_aie4_eos);

#if defined(FLM_ENABLE_CORELIB_AIE4)
    // The largest total the AIE4 path can actually reach: MAX_L capped by the
    // token attention kernel's window, which is one below the prefill path's.
    // Admission and the decode loop both use this, never MAX_L, because the
    // step MAX_L would allow fails past the irrevocable boundary and takes the
    // process with it.
    size_t aie4_active_cap() const;
    void validate_aie4_capacity(
        size_t rendered_tokens,
        std::optional<int> requested_max_new_tokens) const;
    void clear_after_corelib_error();
    std::string generate_aie4(
        chat_meta_info_t& meta_info,
        int length_limit,
        std::ostream& os,
        std::function<bool()> is_cancelled);
    const flm::phi4::Phi4Aie4Metrics& aie4_metrics() const;

    bool uses_corelib_aie4_ = false;
    std::shared_ptr<flm::corelib::CorelibRuntime>
        corelib_runtime_;
    flm::phi4::ForcedContinuationRoute
        forced_continuation_route_ =
            flm::phi4::ForcedContinuationRoute::Automatic;
    std::optional<flm::phi4::ContinuationRoute>
        last_continuation_route_;
    std::uint64_t last_continuation_ns_ = 0;
    std::uint64_t append_continuation_ns_ = 0;
    std::uint64_t reprefill_continuation_ns_ = 0;
#if defined(FLM_CORELIB_TESTING)
    std::optional<flm::phi4::Phi4Aie4Metrics>
        metrics_for_testing_;
#endif
#endif

#if defined(FLM_CORELIB_TESTING)
    using EngineFactoryForTesting =
        std::function<std::unique_ptr<causal_lm>(
            bool,
            const LM_Config&,
            npu_xclbin_manager*,
            const std::filesystem::path&,
            std::uint32_t)>;
    static EngineFactoryForTesting engine_factory_for_testing_;
    friend class flm::phi4::testing::Phi4FrontendTestAccess;
#endif

public:
    Phi4(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool uses_corelib_aie4() const noexcept override {
#if defined(FLM_ENABLE_CORELIB_AIE4)
        return uses_corelib_aie4_;
#else
        return false;
#endif
    }
    //void toggle_enable_think() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    void set_max_length(unsigned int MAX_L) override;
    std::string show_profile() override;
};
