# Adding a New Model to FastFlowLM

This document lists every step required to bring a new model online in FastFlowLM,
from the prebuilt NPU engine down to a standalone test harness.

`Gemma4-12B` is used as the worked example throughout. Substitute:

| Placeholder      | Gemma4-12B example      | Where it is used                                    |
| ---------------- | ----------------------- | --------------------------------------------------- |
| `<model>`        | `gemma4_12b`            | file / directory / engine-library names (snake_case) |
| `<Model>`        | `Gemma4_12B`            | the `AutoModel` subclass name                        |
| `<family>`       | `gemma4-12b`            | model-list family key, CLI tag prefix (kebab-case)   |
| `<ModelName>`    | `Gemma4-12B-IT-NPU2`    | HuggingFace repo, model folder, **and** xclbin folder |

All paths below are relative to `src/` unless stated otherwise.

---

## Architecture in one paragraph

Every model has two layers:

1. **The engine** — `class <model>_npu : public causal_lm`. This is the NPU kernel
   graph (PIMPL-hidden) and is shipped as a *prebuilt* shared library
   (`lib<model>_npu.so` / `<model>_npu.dll`). You normally do **not** write this; it
   comes from the kernel/IRON project.
2. **The AutoModel wrapper** — `class <Model> : public AutoModel`. This is what you
   write. It owns the tokenizer, sampler, chat template, prompt cache and the
   streaming / tool-call parsers, and drives the engine through
   `prefill()` / `forward()` / `checkpoint()` / `restore()`.

---

## Step 0 — Prerequisites (produced outside this repo)

Before writing any wrapper code, these artifacts must exist:

- **Engine header** — `include/models/<model>/<model>_npu.hpp`
  Declares `class <model>_npu : public causal_lm` plus any model-specific payload
  structs (image / audio descriptors for omni models).
- **Engine library**
  - Linux: `lib/xrt/lib<model>_npu.so` and, if the HRX runtime is used,
    `lib/hrx/lib<model>_npu.so`
  - Windows: `lib/<model>_npu.dll` + `<model>_npu.lib`
- **Kernel binaries** — `xclbins/<ModelName>/`
  The folder name must exactly match `name` in `model_list.json` (Step 6).
- **Weights** — a `model.q4nx` plus `config.json`, `tokenizer.json`,
  `tokenizer_config.json` and (if separate) `chat_template.jinja`.

Sanity-check that the `.so` really exports what the header promises:

```bash
nm -DC src/lib/xrt/libgemma4_12b_npu.so | grep gemma4_12b_npu
```

> `_shared_setup_tokenizer()` hard-exits if `tokenizer_config.json` is missing, and
> the chat template must be resolvable either from `chat_template.jinja` or from the
> `chat_template` field inside `tokenizer_config.json`.

---

## Step 1 — Write the AutoModel header

Create `include/AutoModel/modeling_<model>.hpp`.

Pick the closest existing model as your template:

| If your model is…                       | Copy from                     |
| --------------------------------------- | ----------------------------- |
| Gemma4-style (thought/tool markers)      | `modeling_gemma4e.hpp`        |
| Qwen3-style (`<think>` tags)             | `modeling_qwen3.hpp`          |
| Harmony / GPT-OSS-style channels         | `modeling_gpt_oss.hpp`        |
| Vision-language                          | `modeling_qwen3vl.hpp`        |

Required overrides (pure virtual in `AutoModel`):

```cpp
bool insert(chat_meta_info_t&, lm_uniform_input_t&, std::function<bool()>) override;
std::string generate(chat_meta_info_t&, int, std::ostream&, std::function<bool()>) override;
std::string generate_with_prompt(chat_meta_info_t&, lm_uniform_input_t&, int, std::ostream&) override;
std::string apply_chat_template(nlohmann::ordered_json& messages,
                                nlohmann::ordered_json tools) override;
```

Commonly overridden (non-pure, have defaults):

```cpp
void load_model(std::string model_path, json model_info,
                int default_context_length, bool enable_preemption) override;
chat_template_type_t get_chat_template_type() override;   // chat_ml | harmony | gemma4
bool configure_parameter(std::string name, const std::any& value) override;
NonStreamResult parse_nstream_content(const std::string response_text) override;
StreamResult parse_stream_content(const std::string content) override;
StreamResult parse_stream_content_final(const std::string content) override;
```

Also declare the private helpers the base class expects you to supply:
`setup_tokenizer(std::string model_path)` and, if you have a streaming parser, a
shared `parse_stream_content_impl(const std::string, bool is_final)`.

---

## Step 2 — Implement the AutoModel source

Create `common/AutoModel/modeling_<model>.cpp`. For multimodal models, split the
extra pipelines into `modeling_<model>_image.cpp` / `_audio.cpp` — the CMake glob
picks them up automatically.

### 2a. Constructor

```cpp
Gemma4_12B::Gemma4_12B(flm_rt::device* npu_device_inst)
    : AutoModel(npu_device_inst, "Gemma4_12B") {}
```

### 2b. `load_model`

Always in this order — the shared helper parses `config.json`, resolves `MAX_L`,
and sets `this->model_path` / `this->lm_config` before you can construct the engine:

```cpp
this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
this->q4nx = std::make_unique<Q4NX>(this->model_path);
this->lm_engine = std::make_unique<gemma4_12b_npu>(*this->lm_config, this->npu.get(), this->MAX_L);
this->lm_engine->load_weights(*this->q4nx);
this->q4nx.reset();                 // free the mmap'd weights immediately
this->lm_engine->clear_context();
this->setup_tokenizer(model_path);
this->sampler.reset();
sampler_config config;              // set the model's recommended defaults here
this->set_sampler(config);
for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) this->profiler_list[i].reset();
```

### 2c. `setup_tokenizer`

Usually a one-liner delegating to `_shared_setup_tokenizer(model_path)`. Add any
model-specific special-token or stop-token registration after it.

### 2d. `apply_chat_template`

Render the minja template and pass runtime toggles through `extra_context`:

```cpp
inputs.extra_context["enable_thinking"]    = this->enable_think;
inputs.extra_context["user_system_prompt"] = this->system_prompt;   // if supported
```

Only pass `tools` when the array is non-empty — some templates branch on
`tools is defined`.

### 2e. `insert`

Tokenize, then handle the prompt cache and delegate to the shared insert:

```cpp
int restore_idx = -1;
auto* engine = dynamic_cast<gemma4_12b_npu*>(this->lm_engine.get());
if (meta_info.restore_allowed) {
    restore_idx        = engine->restore();
    this->total_tokens = restore_idx;
    this->token_history = checkpoint_his;
}
bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);
checkpoint_his = token_history;
engine->checkpoint();
return success;
```

`_shared_insert` prefix-matches against `token_history` and clears the engine
context on a mismatch, so `checkpoint_his` must be kept in lock-step with
`token_history` or the cache will silently desynchronize.

**Text-only wrapper for an omni model:** warn and drop `input.images` /
`input.audios`, and for REST-style `input.messages` strip the `images` / `audios`
keys per message rather than failing.

### 2f. `generate` / `generate_with_prompt`

`generate` calls `_shared_generate(...)`, then re-checkpoints so the next turn can
restore. `generate_with_prompt` is just `insert()` → optional `"<think>\n"` banner →
`generate()`.

### 2g. Stream & tool-call parsers

Map the model's marker vocabulary to `StreamResult` / `NonStreamResult`. Gemma4, for
example, uses `<|channel>thought` / `<channel|>` for reasoning and
`<|tool_call>` / `<tool_call|>` for tool calls, with relaxed-JSON arguments delimited
by `<|"|>` that need a normalizing rewrite pass before `nlohmann::json::parse`.

---

## Step 3 — Expose the engine header

`include/AutoModel/automodel.hpp` — add the engine include next to the others:

```cpp
#include "models/gemma4e/gemma4e_npu.hpp"
#include "models/gemma4_12b/gemma4_12b_npu.hpp"
```

> Watch for symbol collisions with sibling engines (e.g. `is_swa_layer` overloads).
> They are fine as long as the parameter types are distinct enums.

---

## Step 4 — Register the model family

`include/AutoModel/all_models.hpp` — **four** edits, all required:

```cpp
// 1. include
#include "modeling_gemma4_12b.hpp"

// 2. enum value
enum class SupportedModelFamily { ... gemma4e, gemma4_12b, gpt_oss, ... };

// 3. string -> enum map. The key MUST equal details.family in model_list.json
{"gemma4-12b", SupportedModelFamily::gemma4_12b},

// 4. factory switch case
case SupportedModelFamily::gemma4_12b:
    auto_chat_engine = std::make_unique<Gemma4_12B>(npu_device_inst);
    break;
```

A mismatch between (3) and `model_list.json` shows up at runtime as
"unsupported model family", not at compile time.

---

## Step 5 — Link the engine library

`CMakeLists.txt` — add the bare library name inside `target_link_libraries(flm PUBLIC ...)`
(around line 420):

```cmake
    gemma4e_npu
    gemma4_12b_npu
    gpt_oss_npu
```

Your new `.cpp` files need **no** CMake change — line 238 globs
`common/*/*.cpp` automatically. The `FLM_USE_HRX` option selects whether the
linker searches `lib/hrx` or `lib/xrt`.

---

## Step 6 — Add the model-list entry

`model_list.json` — add a family block keyed by `<family>`, with one entry per size
variant. The CLI tag is `<family>:<variant>` (e.g. `gemma4-12b:12b`).

```json
"gemma4-12b": {
  "12b": {
    "name": "Gemma4-12B-IT-NPU2",
    "url": "https://huggingface.co/FastFlowLM/Gemma4-12B-IT-NPU2",
    "file_url": "https://huggingface.co/api/models/FastFlowLM/Gemma4-12B-IT-NPU2/tree/main",
    "ms_url": "https://modelscope.cn/models/amd/Gemma4-12B-IT-NPU2",
    "size": 12000000000,
    "flm_min_version": "0.9.45",
    "files": ["config.json", "model.q4nx", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja"],
    "vlm": false,
    "asr": false,
    "default_context_length": 32768,
    "max_prefill_len": 4096,
    "details": {
      "format": "NPU2",
      "family": "gemma4-12b",
      "think": true,
      "parameter_size": "12B",
      "quantization_level": "Q4_1"
    },
    "label": ["reasoning", "tool-calling"],
    "footprint": 10.5
  }
}
```

Field notes:

- `name` — **must match both** the on-disk weights folder and `xclbins/<ModelName>/`.
- `details.family` — must match the `modelFamilyMap` key from Step 4.
- `files` — the minimum set `flm pull` must fetch; a missing entry surfaces as a
  tokenizer or config load failure at first run.
- `vlm` / `asr` — capability flags read by the runner to enable image / audio CLI paths.
- `flm_min_version` — refuses to load on older FLM builds.
- `footprint` — GiB of memory reported to the user for model-fits checks.

Model resolution at runtime:
`get_models_directory()` = `$FLM_MODEL_PATH` or `~/flm`;
`get_model_path(tag)` = `<models_dir>/<model_path>/<name>`.

---

## Step 7 — Add the download manifest (needed for `flm pull`)

`model_info.json` — a HuggingFace file listing keyed by the full tag
(`"gemma4-12b:12b"`), containing one object per file with `type`, `oid`, `size`,
`path`, and an `lfs` block for large files. Consumed by
`pull/model_downloader.cpp` and installed by `CMakeLists.txt:878`.

You can regenerate the block from the HF tree API:

```bash
curl -s https://huggingface.co/api/models/FastFlowLM/Gemma4-12B-IT-NPU2/tree/main
```

Skipping this step is fine for local development (point `FLM_MODEL_PATH` at the
weights yourself), but `flm pull <family>:<variant>` will not work without it.

---

## Step 8 — Optional integration points

Only touch these if the model needs them.

| Concern                          | File / location                                                                 |
| -------------------------------- | ------------------------------------------------------------------------------- |
| Audio / vision CLI plumbing      | `runner/runner.cpp` (`asr_supported` at :72, plus :107, :296) and `runner/runner.hpp:54` |
| Template-specific tool-response rewriting | `server/rest_handler.cpp` — see `convert_tool_responses_gemma4()` at :210 and the family dispatch at :1140 |
| REST prompt-cache validation     | `server/rest_handler.cpp:1124` uses `get_chat_template_type()` — return the right `chat_template_type_t` in Step 1 |
| Windows installer (Inno)         | `inno/flm.iss` — add `Source: "<model>_npu.dll"; DestDir: "{app}"; Flags: ignoreversion` |
| Windows installer (WiX)          | `wix/flm.wxs` — add a `<Component>` / `<File>` pair for `<model>_npu.dll`        |
| Public documentation             | `docs/docs/models/<family>.md`, `docs/docs/benchmarks/`, and the model tables in `docs/docs/instructions/server/` |

---

## Step 9 — Create the test harness

Create `test/<model>/` with four files. Copy the closest sibling directory and
rename; the structure is fixed.

### `test/<model>/test.cpp`

Boost `program_options` front-end. Minimal skeleton:

```cpp
flm_rt::device npu_device_global;

std::string exe_dir   = utils::get_executable_directory();
std::string model_dir = utils::get_models_directory();
model_list model_list(exe_dir + "/model_list.json", model_dir);

std::string model_path = model_list.get_model_path(tag);
nlohmann::json model_info = model_list.get_model_info(tag).second;

std::unique_ptr<AutoModel> chat = std::make_unique<Gemma4_12B>(&npu_device_global);
npu_device_global = flm_rt::device(0);        // AFTER construction
chat->load_model(model_path, model_info, -1, preemption);
chat->set_topk(1);                            // deterministic output for testing
chat->configure_parameter("enable_think", enable_think);
```

A useful three-phase short test: first prompt → follow-up on the same context
(exercises the prompt cache) → `clear_context()` → fresh prompt. Wrap each with
`start_total_timer()` / `stop_total_timer()` / `show_profile()`.

### `test/<model>/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.22)
project(gemma4_12b VERSION 1.0.0 LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists.txt)
npu_test_setup()

add_npu_test(test_gemma4_12b test/gemma4_12b
    USE_AUTOMODEL USE_TOKENIZER USE_SAMPLER
    SOURCES "${CMAKE_SOURCE_DIR}/../../common/AutoModel/modeling_gemma4_12b.cpp")

target_compile_definitions(test_gemma4_12b PUBLIC __FLM_VERSION__="0.9.45")
target_link_libraries(test_gemma4_12b PUBLIC gemma4_12b_npu xrt_coreutil)

add_custom_target(test_gemma4_12b_target DEPENDS test_gemma4_12b)
```

### `test/<model>/Makefile`

Linux path. Starts with `-include ../common.mk` (which sets
`BUILD_DIR := ../../build/test/$(CURRENT_DIR)`), then:

```make
SOURCES += test.cpp
SOURCES += ../../common/AutoModel/automodel.cpp
SOURCES += ../../common/AutoModel/modeling_gemma4_12b.cpp
SOURCES += ../../common/tokenizer/tokenizer.cpp
SOURCES += ../../common/modules/sampler.cpp

LDFLAGS += -lgemma4_12b_npu
```

The `test` target copies `model_list.json` into `BUILD_DIR` before running — the
executable looks for it next to itself.

### `test/<model>/activate.sh` (optional convenience)

```bash
#!/usr/bin/bash
export FLM_MODEL_PATH="/scratch/michyu"
cp /path/to/kernels/build/lib/libgemma4_12b_npu.so ../../lib/xrt
# cp -r /path/to/xclbins/Gemma4-12B-IT-NPU2 ../../xclbins/
```

---

## Step 10 — Build and run

```bash
cd src/test/gemma4_12b
source activate.sh          # sets FLM_MODEL_PATH, stages the .so / xclbins
make                        # build only
make test                   # build + run gemma4-12b:12b with the short prompt set
```

`FLM_MODEL_PATH` must point at the **parent** of `models/`, so the weights end up at
`$FLM_MODEL_PATH/models/<ModelName>/`.

Then build the full `flm` binary and smoke-test the real entry points:

```bash
flm run gemma4-12b:12b
flm serve gemma4-12b:12b     # then exercise /v1/chat/completions, streaming, and tools
```

---

## Checklist

- [ ] `include/models/<model>/<model>_npu.hpp` present, exports verified with `nm -DC`
- [ ] `lib/xrt/lib<model>_npu.so` (and `lib/hrx/`, `.dll` + `.lib` for Windows)
- [ ] `xclbins/<ModelName>/` present, name matches `model_list.json`
- [ ] `include/AutoModel/modeling_<model>.hpp`
- [ ] `common/AutoModel/modeling_<model>.cpp`
- [ ] Engine include added to `include/AutoModel/automodel.hpp`
- [ ] All four edits in `include/AutoModel/all_models.hpp`
- [ ] Engine lib added to `target_link_libraries(flm PUBLIC …)` in `CMakeLists.txt`
- [ ] `model_list.json` entry, `details.family` matches `modelFamilyMap`
- [ ] `model_info.json` manifest (if `flm pull` support is wanted)
- [ ] `get_chat_template_type()` returns the correct value for REST prompt caching
- [ ] Installer manifests updated (`inno/flm.iss`, `wix/flm.wxs`) for Windows releases
- [ ] `test/<model>/` builds and runs; prompt-cache turn produces sane output
- [ ] Docs updated under `docs/docs/models/`
