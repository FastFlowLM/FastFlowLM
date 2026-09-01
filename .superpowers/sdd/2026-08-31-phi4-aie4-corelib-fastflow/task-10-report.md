# Task 10 Report: REST, CLI, validation, and shutdown

## Status

Implemented, verified, self-reviewed, and committed locally. Nothing was
pushed.

## Implementation

- Added handler-specific generation-limit parsing. `/api/generate` and
  `/v1/completions` recognize only `max_tokens`; `/v1/chat/completions`
  prefers `max_tokens` and then `max_completion_tokens`.
- Preserved the legacy omitted limit of 4096 while passing `-1` to the AIE4
  generation loop and leaving `requested_max_new_tokens` empty. Explicit
  supported fields are propagated before insertion.
- Kept Ollama `/api/chat` `options.num_predict` solely as a soft generation
  loop bound. The streaming path now calls `generate` rather than performing a
  duplicate insertion, and never reserves AIE4 output capacity.
- Added nested `ModelRequestError` serialization with code, type, and
  `session_cleared`. Nested 400 and 500 responses map to HTTP 400 and 500.
  Streaming failures after output begins send a final nested error chunk;
  failures before the first chunk remain ordinary HTTP errors.
- Typed 400/false errors preserve REST and CLI session state. Typed 500/true
  errors report the frontend clear without calling `clear_context` again.
- Runner propagates positive `/set gen-lim` values to
  `requested_max_new_tokens`; unbounded/nonpositive values remain empty.
  Invalid `/set ctx-len` reports and returns to the prompt. Recoverable
  session-cleared failures print the required exact notice.
- Runner model-load exceptions now unwind to `main` instead of exiting, so
  owned models are destroyed before explicit process shutdown.
- Split the existing `NPUAccessManager` implementation into a lightweight,
  testable source without adding another production mutex. Added the missing
  `/v1/completions` gate and two forced interleaving tests spanning insert and
  generate.
- `flm validate` now reports legacy XDNA2 and optional corelib AIE4 readiness
  independently. AIE4 readiness is obtained through `CorelibRuntime`, which
  checks DLL loading, dependency self-test, device context, and writable fatal
  logging without applying `__NPU_VERSION__`.
- Added best-effort prior fatal-record draining, SIGINT command shutdown, and
  explicit `CorelibRuntime::ShutdownProcess()` after Runner/WebServer
  destruction and on error paths. Validation also shuts down its runtime.
- Feature-OFF main/Runner/REST/server compile checks do not receive corelib
  headers or `FLM_ENABLE_CORELIB_AIE4`.

## RED evidence

- The first `test_generation_limit` build failed with MSVC C1083 because
  `server/generation_limit.hpp` did not exist.

## Verification

- Focused generation-limit, frontend ON/OFF, fatal-record/runtime, and gate
  tests passed.
- Feature-OFF and feature-ON production compile-check targets passed for the
  changed generation, gate, REST, server, Runner, and main translation units.
- Complete standalone Release CTest: 10/10 passed, 0 failed.
- CRLF-aware staged whitespace check passed; the post-commit worktree is clean
  and `git diff --check` passes.
- IDE diagnostics reported no errors.
- Broad `flm` build remains blocked before FastFlow sources by the existing
  tokenizer custom target (`tokenizers_c.lib`: `no such file or directory`,
  Cargo unavailable).

## Self-review

- Rechecked every Task 10 brief item, handler field precedence, omitted and
  explicit lowered-cap behavior, pre/post-stream error delivery, clear
  ownership, gate duration, validation independence, feature guards, and
  destruction-before-shutdown ordering.
- No catalog, installer, or parked-minor files were changed.

## Concerns

- Full HTTP socket-level and interactive CLI execution still require the broad
  product dependency set; focused policy, frontend, runtime, and production
  translation-unit checks cover the behavior available without it.
- Real AIE4 hardware/package validation remains deferred to the integration
  tasks. Existing XRT `NOMINMAX` and duration-conversion warnings remain.

## Commit

- `2fb9302e fix: enforce Phi-4 AIE4 request limits`
- Push: not performed

## Fix round 1/5

### Important findings resolved

- OpenAI chat/completion post-output `ModelRequestError` delivery now writes
  exactly `data: <error-json>\n\n` with `is_final=false`, then
  `data: [DONE]\n\n` with `is_final=true`. Ollama streaming remains NDJSON.
  The byte-exact callback test parses the transmitted SSE data and observes
  `error.session_cleared=true`.
- SIGINT registration is scoped to `serve` and restores the prior handler.
  Non-serve commands retain their ordinary first-Ctrl+C behavior. The tested
  serve shutdown sequence stops admission and joins synchronous handlers,
  destroys WebServer/routes/RestHandler/AutoModel, then performs healthy
  `CorelibRuntime::ShutdownProcess`.
- NPU queue advancement was removed from response callbacks. A single
  scope-safe completion guard in `process_task` advances/releases the gate
  only after the handler and all trailing context/cache work return, including
  parse failures and typed, standard, unknown, cancellation, and final paths.
  The forced interleaving test covers nonstream, streaming, exception, and
  cancellation-final exits and verifies no early second insert or gate leak.
- Omitted-limit routing now queries `AutoModel::uses_corelib_aie4()` only after
  successful model load. The default is false; Phi4 reports true only for its
  actual loaded corelib backend. Tests prove misleading non-Phi metadata and
  feature-OFF/legacy Phi4 retain 4096 while loaded AIE4 uses `-1`.

### RED evidence

- `test_generation_limit` first failed with C1083 for the not-yet-created
  `server/serve_lifecycle.hpp`.
- `test_phi4_frontend_off` first failed because
  `AutoModel/Phi4::uses_corelib_aie4()` did not exist.

### Verification and self-review

- Focused Release CTest passed: generation/gate/lifecycle plus frontend
  OFF/ON, 3/3.
- Complete Release build passed, including production and frontend
  feature-OFF/ON compile checks.
- Complete Release CTest passed: 10/10, 0 failed.
- IDE diagnostics reported no errors. Review confirmed callback releases are
  gone, all three limit decisions occur after load, SIGINT is serve-only, and
  both OpenAI typed streaming catches use the shared SSE sender.
- Unrelated Minor findings were not changed. Existing broad-build dependency
  and hardware-validation concerns remain as documented above.
