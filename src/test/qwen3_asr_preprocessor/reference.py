#!/usr/bin/env python3
"""Compare the C++ Qwen3-ASR frontend with the Hugging Face reference math."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from transformers.audio_utils import mel_filter_bank


def make_waveform(num_samples: int = 16000) -> np.ndarray:
    values = np.empty(num_samples, dtype=np.float32)
    state = 0x12345678
    for index in range(num_samples):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        unit = np.float32(state >> 8) / np.float32(16777216.0)
        values[index] = unit * np.float32(0.4) - np.float32(0.2)
    return values


def reference_features(waveform: np.ndarray) -> np.ndarray:
    waveform_t = torch.from_numpy(waveform).to(torch.float32)
    window = torch.hann_window(400)
    stft = torch.stft(waveform_t, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filter_bank(
        num_frequency_bins=201,
        num_mel_filters=128,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    mel_spec = torch.from_numpy(filters).to(torch.float32).T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).numpy()


def read_cpp_features(path: Path) -> np.ndarray:
    data = path.read_bytes()
    mel_bins, frames, valid_frames = struct.unpack_from("<iii", data)
    values = np.frombuffer(data, dtype="<f4", offset=12).reshape(mel_bins, frames)
    if valid_frames != frames:
        values = values[:, :valid_frames]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path, help="file emitted by test_qwen3_asr_preprocessor --dump")
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()

    expected = reference_features(make_waveform())
    actual = read_cpp_features(args.fixture)
    if actual.shape != expected.shape:
        raise SystemExit(f"shape mismatch: C++={actual.shape}, reference={expected.shape}")

    difference = np.abs(actual - expected)
    print(f"shape: {actual.shape}")
    print(f"max abs error: {difference.max():.8g}")
    print(f"mean abs error: {difference.mean():.8g}")
    np.testing.assert_allclose(actual, expected, atol=args.atol, rtol=args.rtol)
    print("Qwen3-ASR C++/Transformers preprocessing parity passed")


if __name__ == "__main__":
    main()
