/// \file modeling_qwen3_5_omni.cpp
/// \brief Qwen3_5_Omni driver: tokenizer/template/sampler + prefill/decode loop.
/// \author FastFlowLM Team

#include "AutoModel/modeling_qwen3_5_omni.hpp"

Qwen3_5_Omni::Qwen3_5_Omni(flm_rt::device* npu_device_inst)
    : AutoModel(npu_device_inst) {}

/// \brief Pull the thinker-scope vision config (with VL fallbacks).
static uint32_t json_u32(const nlohmann::json& j, const char* key, uint32_t dflt) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<uint32_t>();
    return dflt;
}


static float json_fp32(const nlohmann::json& j, const char* key, float dflt) {
    if (j.contains(key) && !j[key].is_null()) return j[key].get<float>();
    return dflt;
}

void Qwen3_5_Omni::setup_tokenizer(std::string model_path) {
    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
    std::ifstream fs_config(tokenizer_config_path, std::ios::in | std::ios::binary);
    if (fs_config.fail()) {
        header_print("ERROR", "Cannot open " << tokenizer_config_path);
        exit(1);
    }
    std::string data_config((std::istreambuf_iterator<char>(fs_config)), std::istreambuf_iterator<char>());
    fs_config.close();
    auto tokenizer_config = nlohmann::json::parse(data_config);

    this->has_bos_token = !tokenizer_config["bos_token"].is_null();

    std::string chat_template_path = model_path + "/chat_template.jinja";
    if (std::filesystem::exists(chat_template_path)) {
        std::ifstream fs_template(chat_template_path, std::ios::in | std::ios::binary);
        std::string chat_template((std::istreambuf_iterator<char>(fs_template)), std::istreambuf_iterator<char>());
        fs_template.close();
        tokenizer_config["chat_template"] = chat_template;
    } else if (tokenizer_config.find("chat_template") == tokenizer_config.end()) {
        header_print("ERROR", "No template file found and no chat_template in tokenizer_config.json");
        exit(1);
    }

    if (tokenizer_config["eos_token"].is_null()) {
        header_print("ERROR", "eos_token is null in tokenizer_config.json");
        exit(1);
    }

    this->chat_tmpl = std::make_unique<minja::chat_template>(
        tokenizer_config["chat_template"],
        this->has_bos_token ? tokenizer_config["bos_token"] : "",
        tokenizer_config["eos_token"]);

    this->bos_token_id = this->has_bos_token ? tokenizer_config["bos_token_id"].get<int>() : -1;
    this->eos_token = tokenizer_config["eos_token"].get<std::string>();
    this->eos_token_ids.clear();
    for (auto& token : tokenizer_config["eos_token_id"]) {
        this->eos_token_ids.push_back(token.get<int>());
    }
    this->user_system_prompt = "";
    this->extra_context["user_system_prompt"] = this->user_system_prompt;
}

void Qwen3_5_Omni::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    if (this->is_model_loaded && this->model_path == model_path) {
        header_print("FLM", "Model already loaded: " << this->model_path);
        return;
    }
    this->model_path = model_path;
    header_print("FLM", "Loading model: " << this->model_path);

    this->lm_config = std::make_unique<LM_Config>();
    this->lm_config->from_pretrained(this->model_path);

    // The omni thinker vocab lives in thinker_config.text_config, so the
    // top-level "vocab_size" is 0. Republish it at the top level so every
    // downstream reader (set_sampler, set_topk, ...) sees the real vocab.
    {
        auto& jc = this->lm_config->_json_config;
        if (jc.contains("thinker_config") && jc["thinker_config"].contains("text_config")) {
            jc["vocab_size"] =
                json_u32(jc["thinker_config"]["text_config"], "vocab_size", this->lm_config->get("vocab_size"));
        }
    }

    if (this->npu_device_inst == nullptr) {
        header_print("ERROR", "NPU device instance is nullptr");
        exit(1);
    }
    this->npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, this->npu_device_inst, enable_preemption);
    this->enable_preemption = enable_preemption;

    if (default_context_length != -1) {
        this->MAX_L = default_context_length;
    } else {
        this->MAX_L = model_info.value("default_context_length", 4096);
    }

    // vision config (thinker_config.vision_config) with VL fallbacks
    const auto& jc = this->lm_config->_json_config;
    if (jc.contains("thinker_config") && jc["thinker_config"].contains("vision_config")) {
        const auto& vc = jc["thinker_config"]["vision_config"];
        const nlohmann::json &vision_config = this->lm_config->_json_config["vision_config"];

        this->patch_size          = json_u32(vc, "patch_size", this->patch_size);
        this->temporal_patch_size = json_u32(vc, "temporal_patch_size", this->temporal_patch_size);


        this->merge_size          = json_u32(vision_config, "merge_size", this->merge_size);
        this->shortest_edge       = json_u32(vision_config, "shortest_edge", this->shortest_edge);
        this->longest_edge        = json_u32(vision_config, "longest_edge", this->longest_edge);
        this->vision_rescale_factor = json_fp32(vision_config, "rescale_factor", this->vision_rescale_factor);
        this->vision_image_mean = json_fp32(vision_config, "rescale_mean", this->vision_image_mean);
        this->vision_image_std = json_fp32(vision_config, "rescale_std", this->vision_image_std);
    }


    this->engine = std::make_unique<qwen3_5_omni>(*this->lm_config, this->npu.get(), this->MAX_L);
    {
        Q4NX q4nx(this->model_path);
        this->engine->load_weights(q4nx);
    }
    this->engine->clear_context();

    this->tokenizer = std::make_unique<Tokenizer>(this->model_path);
    this->setup_tokenizer(this->model_path);

    // default greedy-ish sampler (overridable via set_* helpers)
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.5f;
    this->set_sampler(config);

    this->is_model_loaded = true;
    this->last_token = -1;
    this->total_tokens = 0;
    this->token_history.clear();
    this->token_history.reserve(this->MAX_L);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }





}

std::string Qwen3_5_Omni::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    if (!tools.empty()) inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Qwen3_5_Omni::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();

    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    qwen3_5_omni_payload_t payload;
    qwen3_5_omni_image_payload_t& image_payload = payload.image_payload;
    qwen3_5_omni_audio_payload_t& audio_payload = payload.audio_payload;
    image_payload.num_images = 0;
    audio_payload.num_audios = 0;

    std::vector<audio_data_t> audio_data_list;


    // ---- build templated text ----
    std::string templated_text;

    if (!input.messages.empty()) { // Server Processing
        int total_images = 0;
        // Build qwen_messages (used for the template) in the same pass that
        // decodes images/audio, so an invalid image (e.g. bad base64) never
        // gets an image-soft-token placeholder; otherwise image_payload would
        // fall out of sync with the placeholders and the prompt would still
        // try to prefill a nonexistent image.
        nlohmann::ordered_json qwen_messages = nlohmann::ordered_json::array();
        for (const auto& message : input.messages) {
            if (!message.contains("images") && !message.contains("audios")) {
                qwen_messages.push_back(message);
                continue;
            }
            nlohmann::ordered_json new_content = nlohmann::ordered_json::array();
            // ---- image preprocessing ----
            if (message.contains("images")) {
                for (auto& img : message["images"]) {
                    std::string img_str = img.get<std::string>();

                    qwen3_5_omni_image_t image = this->load_image_base64(img_str);
                    if (image.width <= 0 || image.height <= 0) {
                        header_print("ERROR", "Skipping invalid base64 image; prefilling language only for this item");
                        continue;
                    }

                    std::vector<int> grid_pair;
                    uint32_t valid_patch_size = 0;
                    uint32_t num_soft_tokens = 0;

                    this->preprocess_image(image, image_payload._processed_pixel_values, grid_pair, valid_patch_size, num_soft_tokens);
                    if (grid_pair.size() < 2 || grid_pair[0] <= 0 || grid_pair[1] <= 0) {
                        header_print("ERROR", "Skipping image that failed to preprocess");
                        continue;
                    }

                    image_payload.image_grid_h_w.push_back(grid_pair);
                    image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
                    image_payload.num_images++;
                    total_images++;

                    new_content.push_back({{"type", "image"}, {"image", img}});
                }
            }
            // ---- audio preprocessing ----
            if (message.contains("audios")) {
                for (auto& aud : message["audios"]) {
                    std::string aud_str = aud.get<std::string>();
                    audio_data_t audio_data = this->load_audio_base64(aud_str, this->audio_sampling_rate, MonoDownmixMode::MEAN);
                    if (audio_data.channels > 1) {
                        header_print("ERROR", "only mono audio is supported");
                        return false;
                    }
                    audio_data_list.push_back(audio_data);
                    new_content.push_back({{"type", "audio"}, {"audio", aud}});
                }
            }
            if(audio_data_list.size()> 0){
                pad_audio(audio_data_list);
                this->extract_spectrogram(audio_data_list, audio_payload);
                for (unsigned int i = 0; i < audio_payload.num_audios; i++) {
                    audio_payload.audio_tokens.push_back(
                        get_audio_tokens(
                            _get_feat_extract_output_length(
                                audio_payload.mel_spectrogram_frames_per_audio[i]
                            )
                        )
                    );
                }
            }

            new_content.push_back({{"type", "text"}, {"text", message.value("content", "")}});
            nlohmann::ordered_json new_item = {
                {"role", message.value("role", "user")},
                {"content", new_content}
            };
            qwen_messages.push_back(new_item);
        }
        header_print("FLM", "Total images: " << total_images);
        templated_text = this->apply_chat_template(qwen_messages, input.tools);
    }
    else { // CLI Processing
        // ---- image preprocessing ----
        for (const auto& img_str : input.images) {
            qwen3_5_omni_image_t image = this->load_image(img_str);
            std::vector<int> grid_pair;
            uint32_t valid_patch_size = 0;
            uint32_t num_soft_tokens = 0;

            this->preprocess_image(image, image_payload._processed_pixel_values, grid_pair, valid_patch_size, num_soft_tokens);

            image_payload.image_grid_h_w.push_back(grid_pair);
            image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
            image_payload.num_images++;
        }

        // ---- audio preprocessing ----
        for (const auto& aud_str : input.audios) {
            audio_data_t audio_data = this->load_audio(aud_str, this->audio_sampling_rate, MonoDownmixMode::MEAN);
            if (audio_data.channels > 1) {
                header_print("ERROR", "only mono audio is supported");
                return false;
            }
          
            audio_data_list.push_back(audio_data);
        }
        if(audio_data_list.size()> 0){
            pad_audio(audio_data_list);
            this->extract_spectrogram(audio_data_list, audio_payload);
            for (unsigned int i = 0; i < audio_payload.num_audios; i++) {
                audio_payload.audio_tokens.push_back(
                    get_audio_tokens(
                        _get_feat_extract_output_length(
                            audio_payload.mel_spectrogram_frames_per_audio[i]
                        )
                    )
                );
            }
        }

        // ---- build templated text (CLI) ----
        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        for (size_t i = 0; i < input.images.size(); i++) {
            content["content"].push_back({{"type", "image"}, {"image", input.images[i]}});
        }
        for (size_t i = 0; i < input.audios.size(); i++) {
            content["content"].push_back({{"type", "audio"}, {"audio", input.audios[i]}});
        }
        content["content"].push_back({{"type", "text"}, {"text", input.prompt}});
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }


    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // ---- expand soft-token placeholders ----
    int total_image_tokens = 0;
    for (unsigned int i = 0; i < image_payload.num_images; i++) {
        total_image_tokens += image_payload.num_soft_tokens_per_image[i];
    }


    std::vector<int> tokens;
    tokens.reserve(tokens_init.size() + total_image_tokens);
   auto total_audio_tokens= 0;



    int image_counter = 0;
    int audio_counter = 0;
    for (size_t i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_token_id) {
            for (unsigned int j = 0; j < image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_token_id);
            }
            image_counter++;
        } else if (tokens_init[i] == audio_token_id) {
            // first, do a tokenizer on the new message
            // insert this into the tokens_init
            std::vector<int> new_audio_tokens = this->tokenizer->encode(audio_payload.audio_tokens[audio_counter]);
            total_audio_tokens+= new_audio_tokens.size();
            tokens.push_back(audio_start_token_id);
            for(size_t j = 0; j < new_audio_tokens.size(); j++){
                tokens.push_back(new_audio_tokens[j]);
            }
            tokens.push_back(audio_end_token_id);
            audio_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // header_print("FLM", "Prompt tokens: " << tokens.size()
    //     << " (images: " << image_payload.num_images << ", soft image tokens: " << total_image_tokens
    //     << "; audios: " << audio_payload.num_audios << ", soft audio tokens: " << total_audio_tokens << ")");

    // ---- prompt-cache prefix matching ----
    // Compute how many leading tokens are already in the cached KV state
    // (checkpoint_his). _shared_insert requires checkpoint_his to be fully
    // matched; otherwise it clears context and skips nothing. We replicate
    // that logic here so we can also trim the multi-modal payload accordingly.
    size_t prefix_skip_count = 0;
    {
        const size_t idx = this->checkpoint_his.size();
        for (size_t i = 0; i < idx; i++) {
            if (i < tokens.size() && tokens[i] == this->checkpoint_his[i]) {
                prefix_skip_count++;
            } else {
                break;
            }
        }
        if (prefix_skip_count != idx) {
            // header_print("KV-Cache", "Checkpoint not mathchedm here!");
            prefix_skip_count = 0;
            // this->engine->clear_context();
            // this->checkpoint_his.clear();
            // this->total_tokens = 0;
            // this->token_history.clear();
        }
    }

    // Drop leading images/audios whose soft tokens fall entirely within the
    // cached prefix so the payload stays aligned with the surviving tokens.
    if (prefix_skip_count > 0 && (image_payload.num_images > 0 || audio_payload.num_audios > 0)) {
        int skipped_image_tokens = 0;
        int skipped_audio_tokens = 0;  // counts audio_token_id soft tokens only (not boa/eoa)
        for (size_t i = 0; i < prefix_skip_count; i++) {
            if (tokens[i] == image_token_id) skipped_image_tokens++;
            else if (tokens[i] == audio_token_id) skipped_audio_tokens++;
            // audio_start/end token ids are not counted; the boa/eoa are handled implicitly
        }

        auto drop_front = [](auto& vec, size_t n) {
            if (n == 0) return;
            if (n >= vec.size()) { vec.clear(); return; }
            vec.erase(vec.begin(), vec.begin() + n);
        };

        size_t images_to_drop = 0;
        {
            int consumed = 0;
            for (unsigned i = 0; i < image_payload.num_images; i++) {
                const int n = static_cast<int>(image_payload.num_soft_tokens_per_image[i]);
                if (consumed + n <= skipped_image_tokens) { consumed += n; images_to_drop++; }
                else break;
            }
        }
        if (images_to_drop > 0) {
            // Compute how many bf16 elements to erase from the flat pixel buffer.
            // Each image contributed grid_h * grid_w * patch_size^2 * temporal_patch_size * 3 values.
            size_t pixel_elems_to_drop = 0;
            for (size_t i = 0; i < images_to_drop; i++) {
                const int grid_h = image_payload.image_grid_h_w[i][0];
                const int grid_w = image_payload.image_grid_h_w[i][1];
                pixel_elems_to_drop += static_cast<size_t>(grid_h) * grid_w
                    * this->patch_size * this->patch_size
                    * this->temporal_patch_size * 3;
            }
            drop_front(image_payload._processed_pixel_values, pixel_elems_to_drop);
            drop_front(image_payload.image_grid_h_w,              images_to_drop);
            drop_front(image_payload.num_soft_tokens_per_image,   images_to_drop);
            image_payload.num_images -= static_cast<unsigned>(images_to_drop);
            header_print("FLM", "Prompt-cache hit: dropped " << images_to_drop << " cached image(s) from payload");
        }

        // Count complete audio blocks (audio_start..audio_end pairs) that fall
        // entirely within the cached prefix. Each such pair maps 1:1 to an audio
        // in the payload. We scan for start/end bracket tokens rather than
        // audio_token_id because the expanded stream replaces the placeholder with
        // encoded timestamp+pad tokens — bare audio_token_id never appears there.
        size_t audios_to_drop = 0;
        {
            bool in_audio_block = false;
            for (size_t i = 0; i < prefix_skip_count; i++) {
                if (tokens[i] == audio_start_token_id) {
                    in_audio_block = true;
                } else if (tokens[i] == audio_end_token_id && in_audio_block) {
                    in_audio_block = false;
                    audios_to_drop++;
                }
            }
            // A block still open at the prefix boundary is only partially cached;
            // do not drop it.
        }
        if (audios_to_drop > 0) {
            // Each audio has its own mel_spectrograms[i] buffer (not a flat concat),
            // so dropping is a simple front-erase on each per-audio vector.
            drop_front(audio_payload.mel_spectrograms,                 audios_to_drop);
            drop_front(audio_payload.mel_spectrogram_frames_per_audio, audios_to_drop);
            drop_front(audio_payload.mel_spectrogram_bins_per_audio,   audios_to_drop);
            drop_front(audio_payload.audio_tokens,                     audios_to_drop);
            audio_payload.num_audios -= static_cast<unsigned>(audios_to_drop);
            header_print("FLM", "Prompt-cache hit: dropped " << audios_to_drop << " cached audio(s) from payload");
        }

    }
    const bool has_modal = (image_payload.num_images > 0) || (audio_payload.num_audios > 0);


    // Restore KV cache from checkpoint if caller allows it
    if (meta_info.restore_allowed) {
        int restore_idx = this->engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = this->checkpoint_his;
    }

    size_t n = tokens.size();
    tokens.resize(n - 4);

    // prefix check for tokens and token history to see if we can skip some tokens
    const size_t idx = this->token_history.size();
    size_t skip_count = 0;
    for (size_t i = 0; i < idx; i++) {
        if (tokens[i] == this->token_history[i]) {
            skip_count++;
        } 
        else {
            break;
        }
    }
    if (skip_count != idx) {
        clear_context();
        skip_count = 0;
    }
    tokens.erase(tokens.begin(), tokens.begin() + skip_count);

    if (this->total_tokens + tokens.size() >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping prefilling...");
        return false;
    }
    for (int token : tokens){
        this->token_history.push_back(token);
    }

    auto prefill_start = this->profiler_list[PREFILL_TIME].start();

    int max_prefill_len = meta_info.max_prefill_len;

    // Index (exclusive) of the last multimodal soft token in the surviving
    // token sequence. The chunker uses this to guarantee the first prefill
    // chunk is large enough to carry the entire multimodal payload.
    int first_len_run = 0;
    for (int i = static_cast<int>(tokens.size()) - 1; i >= 0; i--) {
        if (tokens[i] == image_token_id || tokens[i] == audio_token_id ||
            tokens[i] == audio_start_token_id || tokens[i] == audio_end_token_id) {
            first_len_run = i + 1;
            break;
        }
    }

    max_prefill_len = 1 << static_cast<int>(std::ceil(std::log2(max_prefill_len)));
    if (max_prefill_len < 512) {
        this->last_thinker_result = this->engine->prefill(tokens, has_modal ? &payload : nullptr);
    }
    else{
        if (first_len_run > 0) {
            int new_max_len = 1 << static_cast<int>(std::ceil(std::log2(first_len_run)));
            if (new_max_len > max_prefill_len) {
                max_prefill_len = new_max_len;
            }
        }
        int chunks = (tokens.size() + max_prefill_len - 1) / max_prefill_len;
        for (int i = 0; i < chunks; i++) {
            if (is_cancelled()) {
                meta_info.stop_reason = CANCEL_DETECTED;
                // reset stream content 
                buffer_.clear();
                current_mode_ = StreamEventType::CONTENT;
                tool_name_.clear();
                is_in_tool_block_ = false;
                break;
            }
            int start = i * max_prefill_len;
            int end = std::min(static_cast<int>(tokens.size()), (i + 1) * max_prefill_len);
            std::vector<int> chunk_tokens(tokens.begin() + start, tokens.begin() + end);
            header_print("FLM", "Prefill chunk " + std::to_string(i+1) + "/" + std::to_string(chunks) + " with " + std::to_string(chunk_tokens.size()) + " tokens");
            auto chunk_thinker_result = this->engine->prefill(chunk_tokens, (i == 0) ? &payload : nullptr);
            if (i == chunks - 1) {
                this->last_thinker_result = chunk_thinker_result;
            }
        }
    }

    auto prefill_end = this->profiler_list[PREFILL_TIME].stop(tokens.size());
    meta_info.prefill_duration = (uint64_t)time_utils::duration_ns(prefill_start, prefill_end).first;
    meta_info.prompt_tokens = tokens.size(); 

    this->total_tokens += tokens.size();

    this->profiler_list[SAMPLING_TIME].start();
    this->last_token = this->sampler->sample(this->last_thinker_result.logits);
    this->profiler_list[SAMPLING_TIME].stop(1);

    this->checkpoint_his = this->token_history;
    this->engine->checkpoint();
    return true;
}

std::string Qwen3_5_Omni::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::vector<int> sampled_tokens;
    std::string result;
    if (length_limit > 0){
        sampled_tokens.reserve(length_limit);
    }
    else{
        sampled_tokens.reserve(4096);
    }
    assert(this->last_token != -1);
    stop_reason_t reason = EOT_DETECTED;

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    std::string token_str;
    int sampled_token;
    int last_sampled_token;

    // manually decoding <think> \n <\think> \n
    {
        this->token_history.push_back(think_start_id);
        this->engine->forward(think_start_id);
        token_str = this->tokenizer->run_time_decoder(think_start_id);

        // \n\n
        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        this->engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        
        this->token_history.push_back(think_end_id);
        this->profiler_list[DECODING_TIME].start();
        this->engine->forward(think_end_id);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(think_end_id);

        this->token_history.push_back(271);
        this->profiler_list[DECODING_TIME].start();
        this->last_thinker_result = this->engine->forward(271);
        this->profiler_list[DECODING_TIME].stop(1);
        token_str = this->tokenizer->run_time_decoder(271);
        sampled_token = this->sampler->sample(this->last_thinker_result.logits);

        this->total_tokens++;
        meta_info.generated_tokens++;
        last_sampled_token = sampled_token;
        token_str = this->tokenizer->run_time_decoder(last_sampled_token);
        result += token_str;
        os << token_str << std::flush;
    }

    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
        reason = MAX_LENGTH_REACHED;
        return result;
    }

    while (this->total_tokens < this->MAX_L) {
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            // reset stream content 
            buffer_.clear();
            current_mode_ = StreamEventType::CONTENT;
            tool_name_.clear();
            is_in_tool_block_ = false;
            break;
        }
        this->profiler_list[DECODING_TIME].start();
        this->last_thinker_result = this->engine->forward(last_sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);

        this->profiler_list[SAMPLING_TIME].start();
        int sampled_token = this->sampler->sample(this->last_thinker_result.logits);
        this->profiler_list[SAMPLING_TIME].stop(1);
        this->total_tokens++;
        last_sampled_token = sampled_token;
 
        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)) { // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        this->token_history.push_back(sampled_token);
        if (this->is_eos(sampled_token)) {
            meta_info.generated_tokens++;
            this->engine->forward(last_sampled_token);
            break;
        }
        meta_info.generated_tokens++;
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)) {
            reason = MAX_LENGTH_REACHED;
            break;
        }
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;
    if (this->total_tokens >= this->MAX_L){
        header_print("WARNING", "Max length reached, stopping generation...");
    }

    std::cout << std::endl;
    header_print("FLM", "Model RAW Output: \n" + result);
    return result;
}

std::string Qwen3_5_Omni::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    header_print("FLM", "Prompt inserted, starting generation...");
    return this->generate(meta_info, length_limit, os);
}

buffer<bf16> Qwen3_5_Omni::say(std::string wav_out_path) {
    try {
        buffer<bf16> audio = this->engine->say(this->last_thinker_result);
        // TODO: once the talker/codec path lands, decode `audio` to PCM and write
        //       a WAV file at wav_out_path.
        if (!wav_out_path.empty()) {
            header_print("FLM", "Audio output produced (" << audio.size()
                << " elements); WAV writing not implemented yet.");
        }
        return audio;
    } catch (const std::exception& e) {
        header_print("FLM", "Audio output not available yet: " << e.what());
        return buffer<bf16>();
    }
}

void Qwen3_5_Omni::set_sampler(sampler_config& sampler_config) {
    if (this->sampler != nullptr) this->sampler.reset();
    this->sampler = std::make_unique<Sampler>((int)this->lm_config->get("vocab_size"), sampler_config);
}

void Qwen3_5_Omni::clear_context() {
    this->total_tokens = 0;
    this->last_token = -1;
    this->token_history.clear();
    this->checkpoint_his.clear();
    this->engine->clear_context();
    this->total_tokens = 0;
    this->sampler->reset_penalties();
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    this->last_prefill_time = { 0, "us" };
}

void Qwen3_5_Omni::set_max_length(unsigned int new_max) {
    this->MAX_L = std::max(new_max, this->MAX_L);
    if (this->engine != nullptr) this->engine->update_max_length(this->MAX_L);
}

int Qwen3_5_Omni::get_current_context_length() {
    return this->total_tokens;
}

std::string Qwen3_5_Omni::show_model_info() {
    try {
        return this->lm_config->_str();
    } catch (const std::exception& e) {
        return "Error showing model info: " + std::string(e.what());
    }
}

std::string Qwen3_5_Omni::show_profile() {
    std::stringstream ss;
    time_utils::time_with_unit time = this->profiler_list[TOTAL_TIME].get_total_time();
    ss << "  Statistics:" << std::endl;
    ss << "    Total tokens:        " << this->get_current_context_length() << std::endl;
    ss << "    Total time:          " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[DECODING_TIME].get_total_time();
    ss << "    Decoding time:       " << time.first << " " << time.second << std::endl;
    time = this->profiler_list[PREFILL_TIME].get_total_time();
    ss << "    Prefill time:        " << time.first << " " << time.second << std::endl;
    ss << "    Average decoding speed:       " << this->profiler_list[DECODING_TIME].get_average_speed() << " tokens/s" << std::endl;
    ss << "    Average prefill  speed:       " << this->profiler_list[PREFILL_TIME].get_average_speed() << " tokens/s" << std::endl;
    return ss.str();
}

std::pair<std::string, std::vector<int>> Qwen3_5_Omni::get_history() {
    std::vector<int> history = this->token_history;
    std::string all_context = this->tokenizer->decode(history);
    return std::make_pair(all_context, history);
}

// Non-stream
NonStreamResult Qwen3_5_Omni::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";
    std::string func_end_tag = "</function>";
    std::string func_open = "<function=";
    std::string param_open = "<parameter=";
    std::string param_close = "</parameter>";

    auto trim_tool_value = [](std::string value) {
        while (!value.empty() && (value.front() == '\n' || value.front() == '\r' || value.front() == ' ' || value.front() == '\t')) {
            value.erase(0, 1);
        }
        while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        return value;
    };

    size_t search_from = 0;

    while (true) {
        size_t start_pos = response_text.find(start_tag, search_from);
        if (start_pos == std::string::npos) break;

        size_t block_content_start = start_pos + start_tag.length();
        size_t end_pos = response_text.find(end_tag, block_content_start);

        size_t block_end;
        if (end_pos != std::string::npos) {
            block_end = end_pos;
            search_from = end_pos + end_tag.length();
        } else {
            // Unclosed tag — search for </function> fallback
            size_t func_end_pos = response_text.find(func_end_tag, block_content_start);
            if (func_end_pos != std::string::npos) {
                block_end = func_end_pos + func_end_tag.length();
            } else {
                block_end = response_text.length();
            }
            search_from = block_end;
        }

        std::string block = response_text.substr(block_content_start, block_end - block_content_start);

        std::string tool_name;
        size_t func_start = block.find(func_open);
        if (func_start != std::string::npos) {
            func_start += func_open.length();
            size_t func_name_end = block.find(">", func_start);
            if (func_name_end != std::string::npos) {
                tool_name = block.substr(func_start, func_name_end - func_start);
            }
        }

        nlohmann::json args = nlohmann::json::object();
        size_t pos = 0;

        while (true) {
            size_t param_start = block.find(param_open, pos);
            if (param_start == std::string::npos) break;

            param_start += param_open.length();
            size_t param_name_end = block.find(">", param_start);
            if (param_name_end == std::string::npos) break;

            std::string param_name = block.substr(param_start, param_name_end - param_start);
            size_t value_start = param_name_end + 1;
            size_t value_end = block.find(param_close, value_start);

            size_t next_param_pos = block.find(param_open, value_start);
            size_t func_boundary_pos = block.find(func_end_tag, value_start);

            auto use_earlier_boundary = [&value_end](size_t boundary_pos) {
                if (boundary_pos != std::string::npos && (value_end == std::string::npos || boundary_pos < value_end)) {
                    value_end = boundary_pos;
                }
            };

            use_earlier_boundary(next_param_pos);
            use_earlier_boundary(func_boundary_pos);

            if (value_end == std::string::npos) {
                value_end = block.length();
            }

            std::string param_value = trim_tool_value(block.substr(value_start, value_end - value_start));

            try {
                args[param_name] = nlohmann::json::parse(param_value);
            }
            catch (...) {
                args[param_name] = param_value;
            }

            pos = value_end;
            if (block.compare(value_end, param_close.length(), param_close) == 0) {
                pos += param_close.length();
            }
        }

        result.tool_calls_list.emplace_back(tool_name, args.dump());
    }

    if (result.tool_calls_list.empty()) {
        result.content = response_text;
    } else {
        // Populate legacy single-tool fields from the first call for backward compatibility
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
        // Extract content before the first <tool_call>
        size_t first_tool = response_text.find(start_tag);
        if (first_tool != std::string::npos && first_tool > 0) {
            result.content = trim_tool_value(response_text.substr(0, first_tool));
        }
    }

    return result;
}

// Stream
StreamResult Qwen3_5_Omni::parse_stream_content(const std::string content) {
    return parse_stream_content_impl(content, false);
}

StreamResult Qwen3_5_Omni::parse_stream_content_final(const std::string content) {
    return parse_stream_content_impl(content, true);
}

StreamResult Qwen3_5_Omni::parse_stream_content_impl(const std::string content, bool is_final) {
    const std::string MARKER_THINK_START = "<think>";
    const std::string MARKER_THINK_END = "</think>";
    const std::string MARKER_TOOL_START = "<tool_call>";
    const std::string MARKER_TOOL_END = "</tool_call>";
    const std::string MARKER_FUNC_END = "</function>";


    StreamResult result;
    buffer_ += content;

    while (true) {
        if (!is_in_tool_block_) {
            size_t stray_end_pos = buffer_.find(MARKER_TOOL_END);
            if (stray_end_pos != std::string::npos) {
                buffer_.erase(stray_end_pos, MARKER_TOOL_END.length());
            }
        }

        if (!is_in_tool_block_) {
            size_t tool_start_pos = buffer_.find(MARKER_TOOL_START);
            if (tool_start_pos != std::string::npos) {
                if (tool_start_pos > 0) {
                    result.content = buffer_.substr(0, tool_start_pos);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(tool_start_pos);
                    return result;
                }

                is_in_tool_block_ = true;
                buffer_ = buffer_.substr(MARKER_TOOL_START.length());
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        // tool calling process
        if (is_in_tool_block_) {
            size_t tool_end_pos = buffer_.find(MARKER_TOOL_END);
            size_t func_end_pos = buffer_.find(MARKER_FUNC_END);

            if (tool_end_pos != std::string::npos || func_end_pos != std::string::npos || (is_final && !buffer_.empty())) {
                size_t actual_end_pos = buffer_.size();
                size_t skip_length = 0;

                if (tool_end_pos != std::string::npos) {
                    actual_end_pos = tool_end_pos;
                    skip_length = MARKER_TOOL_END.length();
                }
                else if (func_end_pos != std::string::npos) {
                    actual_end_pos = func_end_pos;
                    skip_length = MARKER_FUNC_END.length();
                }

                std::string block = buffer_.substr(0, actual_end_pos + skip_length);
                buffer_ = buffer_.substr(actual_end_pos + skip_length);
                is_in_tool_block_ = false;

                try {
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));

                    // parse function name
                    std::string func_open = "<function=";
                    size_t func_start = block.find(func_open);
                    if (func_start != std::string::npos) {
                        func_start += func_open.length();
                        size_t func_end = block.find(">", func_start);
                        if (func_end != std::string::npos) {
                            result.tool_name = block.substr(func_start, func_end - func_start);
                        }
                    }

                    // parse parameters
                    nlohmann::json args = nlohmann::json::object();
                    std::string param_open = "<parameter=";
                    std::string param_close = "</parameter>";
                    size_t search_pos = 0;

                    while (true) {
                        size_t p_start = block.find(param_open, search_pos);
                        if (p_start == std::string::npos) break;
                        p_start += param_open.length();
                        size_t p_name_end = block.find(">", p_start);
                        if (p_name_end == std::string::npos) break;
                        std::string param_name = block.substr(p_start, p_name_end - p_start);

                        size_t val_start = p_name_end + 1;
                        if (val_start < block.size() && block[val_start] == '\n') val_start++;

                        size_t param_close_pos = block.find(param_close, val_start);
                        size_t val_end = param_close_pos;

                        size_t next_param_pos = block.find(param_open, val_start);
                        size_t func_boundary_pos = block.find(MARKER_FUNC_END, val_start);
                        size_t tool_boundary_pos = block.find(MARKER_TOOL_END, val_start);

                        auto use_earlier_boundary = [&val_end](size_t boundary_pos) {
                            if (boundary_pos != std::string::npos && (val_end == std::string::npos || boundary_pos < val_end)) {
                                val_end = boundary_pos;
                            }
                        };

                        use_earlier_boundary(next_param_pos);
                        use_earlier_boundary(func_boundary_pos);
                        use_earlier_boundary(tool_boundary_pos);

                        if (val_end == std::string::npos && is_final) {
                            val_end = block.size();
                        }
                        if (val_end == std::string::npos) break;

                        std::string param_value = block.substr(val_start, val_end - val_start);

                        // Enhanced trim: handle multiple newlines or spaces that the model may generate after a parameter
                        while(!param_value.empty() && (param_value.back() == '\n' || param_value.back() == '\r' || param_value.back() == ' ')) {
                            param_value.pop_back();
                        }

                        try {
                            // Try to parse as native JSON type (Integer, Float, Boolean, Array, Object)
                            args[param_name] = nlohmann::json::parse(param_value);
                        }
                        catch (...) {
                            args[param_name] = param_value;
                        }

                        search_pos = param_close_pos != std::string::npos && val_end == param_close_pos
                            ? val_end + param_close.length()
                            : val_end;
                    }
                    result.tool_args_str = args.dump();
                    return result;
                }
                catch (...) {
                    result.type = StreamEventType::CONTENT;
                    result.content = "[Error parsing tool call]";
                    return result;
                }
            }
            else {
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        if (current_mode_ == StreamEventType::CONTENT) {
            size_t think_start_pos = buffer_.find(MARKER_THINK_START);
            if (think_start_pos != std::string::npos) {
                if (think_start_pos > 0) {
                    result.content = buffer_.substr(0, think_start_pos);
                    result.type = StreamEventType::CONTENT;
                    buffer_ = buffer_.substr(think_start_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_START.length());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        }
        else if (current_mode_ == StreamEventType::REASONING) {
            size_t think_end_pos = buffer_.find(MARKER_THINK_END);
            if (think_end_pos != std::string::npos) {
                if (think_end_pos > 0) {
                    result.content = buffer_.substr(0, think_end_pos);
                    result.type = StreamEventType::REASONING;
                    buffer_ = buffer_.substr(think_end_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_END.length());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }

        if (!buffer_.empty()) {
            size_t last_lt = buffer_.rfind('<');
            // If '<' appears at the end (possibly an incomplete <tool_call> or <think> tag)
            if (last_lt != std::string::npos && (buffer_.length() - last_lt) <= 15) {
                if (last_lt > 0) {
                    // Only output the content before '<'
                    result.content = buffer_.substr(0, last_lt);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(last_lt);
                    return result;
                } else {
                    // If '<' is the first character in the buffer, directly wait for the next chunk
                    result.type = StreamEventType::WAITING;
                    return result;
                }
            }

            result.content = buffer_;
            result.type = current_mode_;
            buffer_.clear();
            return result;
        }

        break;
    }

    result.type = current_mode_;
    return result;
}