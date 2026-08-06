# Qwen3-ASR preprocessor parity test

This standalone test validates the C++ waveform frontend without requiring a
Qwen3-ASR NPU kernel or model weights.

It checks the Qwen3-ASR feature-length rules, minimum input padding, mel-axis
padding, tensor layout, and finite outputs. The Python script then compares the
C++ features with the Hugging Face reference operations.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure

./build/test_qwen3_asr_preprocessor --dump /tmp/qwen3_asr_features.bin
python3 reference.py /tmp/qwen3_asr_features.bin
```

The Python parity step requires `numpy`, `torch`, and `transformers`.
