from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from tools import generate_phi4_corelib_manifest as manifest_tool


UPSTREAM_COMMIT = "e751fb68c2cfffe6b0d32942118f75ac0a0365bb"
FLM_MIN_VERSION = "1.0.4"
EXPECTED_EOS_IDS = [200020, 199999]
UPSTREAM_REPOSITORY = (
    "https://huggingface.co/amd/phi-4-mini-instruct-oga-dml"
)
UPSTREAM_API_URL = (
    "https://huggingface.co/api/models/amd/"
    f"phi-4-mini-instruct-oga-dml/tree/{UPSTREAM_COMMIT}"
    "?recursive=true&expand=false"
)

_EXPECTED_DECODER = {
    "head_size": 128,
    "hidden_size": 3072,
    "num_attention_heads": 24,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
}
_EXPECTED_MODEL = {
    "vocab_size": 200064,
}


def _require_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_exact_value(
    mapping: dict[str, object],
    field: str,
    expected: object,
    context: str,
) -> object:
    value = mapping.get(field)
    if value != expected:
        raise ValueError(
            f"{context}.{field} must be {expected!r}, got {value!r}"
        )
    return value


def normalize_config(genai_config: dict[str, object]) -> dict[str, object]:
    model = _require_mapping(genai_config.get("model"), "model")
    decoder = _require_mapping(model.get("decoder"), "model.decoder")
    for field, expected in _EXPECTED_DECODER.items():
        _require_exact_value(decoder, field, expected, "model.decoder")
    for field, expected in _EXPECTED_MODEL.items():
        _require_exact_value(model, field, expected, "model")

    return {
        "flm_version": FLM_MIN_VERSION,
        "head_dim": decoder["head_size"],
        "hidden_size": decoder["hidden_size"],
        "intermediate_size": 8192,
        "model_type": "phi4",
        "num_attention_heads": decoder["num_attention_heads"],
        "num_hidden_layers": decoder["num_hidden_layers"],
        "num_key_value_heads": decoder["num_key_value_heads"],
        "rms_norm_eps": 1.0e-5,
        "vocab_size": model["vocab_size"],
    }


def normalize_tokenizer_config(
    tokenizer_config: dict[str, object],
    chat_template: str,
    genai_config: dict[str, object],
) -> dict[str, object]:
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("chat template must be a non-empty string")
    model = _require_mapping(genai_config.get("model"), "model")
    _require_exact_value(
        model,
        "eos_token_id",
        EXPECTED_EOS_IDS,
        "model",
    )
    normalized = dict(tokenizer_config)
    normalized["chat_template"] = chat_template
    normalized["eos_token_id"] = list(EXPECTED_EOS_IDS)
    return normalized


def catalog_measurements(
    model_dir: Path,
    logical_sizes: dict[str, int] | None = None,
) -> tuple[int, float]:
    model_dir = Path(model_dir)
    logical_sizes = {} if logical_sizes is None else dict(logical_sizes)
    files = [path for path in model_dir.rglob("*") if path.is_file()]
    relative_paths = {
        path.relative_to(model_dir).as_posix(): path for path in files
    }
    unknown = sorted(set(logical_sizes) - set(relative_paths))
    if unknown:
        raise ValueError(
            f"logical size has no matching package file: {unknown[0]}"
        )
    size = 0
    for relative, path in relative_paths.items():
        logical_size = logical_sizes.get(relative, path.stat().st_size)
        if (
            isinstance(logical_size, bool)
            or not isinstance(logical_size, int)
            or logical_size < 0
        ):
            raise ValueError(f"invalid logical size for {relative}")
        size += logical_size
    footprint_gib = round(size / (1024**3), 2)
    return size, footprint_gib


def _sha256_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_provenance(
    upstream_dir: Path,
    overlay_dir: Path,
    upstream_commit: str,
    generated_files: list[str],
    git_files: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if upstream_commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream commit must be pinned to {UPSTREAM_COMMIT}"
        )
    upstream_dir = Path(upstream_dir)
    overlay_dir = Path(overlay_dir)
    input_names = (
        "chat_template.jinja",
        "genai_config.json",
        "tokenizer_config.json",
    )
    provenance: dict[str, object] = {
        "upstream": {
            "repository": (
                UPSTREAM_REPOSITORY
            ),
            "commit": upstream_commit,
            "inputs": {
                name: _sha256_record(upstream_dir / name)
                for name in input_names
            },
        },
        "generated": {
            name: _sha256_record(overlay_dir / name)
            for name in sorted(generated_files)
        },
    }
    if git_files is not None:
        provenance["upstream"]["git_files"] = sorted(
            git_files,
            key=lambda record: record["path"],
        )
    return provenance


def _git_blob_oid(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _index_git_records(
    records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Git metadata record must be an object")
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("Git metadata record has an invalid path")
        if path in indexed:
            raise ValueError(f"duplicate Git metadata path: {path}")
        if record.get("type") != "file":
            raise ValueError(f"Git metadata path is not a file: {path}")
        indexed[path] = record
    return indexed


def _validate_input_git_record(
    upstream_dir: Path,
    record: dict[str, object],
) -> None:
    path = upstream_dir / str(record["path"])
    data = path.read_bytes()
    if record.get("size") != len(data):
        raise ValueError(f"Git metadata size does not match {record['path']}")
    if "lfs" in record:
        lfs = _require_mapping(record["lfs"], f"{record['path']}.lfs")
        if lfs.get("size") != len(data):
            raise ValueError(
                f"Git LFS size does not match {record['path']}"
            )
        if lfs.get("oid") != hashlib.sha256(data).hexdigest():
            raise ValueError(
                f"Git LFS SHA-256 does not match {record['path']}"
            )
    elif record.get("oid") != _git_blob_oid(data):
        raise ValueError(f"Git blob OID does not match {record['path']}")


def _manifest_file_metadata(
    indexed: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for name in ("model.onnx", "model.onnx.data"):
        record = indexed.get(name)
        if record is None:
            raise ValueError(f"Git metadata is missing {name}")
        lfs = _require_mapping(record.get("lfs"), f"{name}.lfs")
        if record.get("size") != lfs.get("size"):
            raise ValueError(f"Git LFS logical size does not match {name}")
        metadata[name] = {
            "size": lfs.get("size"),
            "sha256": lfs.get("oid"),
        }
    return metadata


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_overlay(
    upstream_dir: Path,
    overlay_dir: Path,
    upstream_commit: str,
    git_files: list[dict[str, object]],
) -> dict[str, object]:
    if upstream_commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream commit must be pinned to {UPSTREAM_COMMIT}"
        )
    upstream_dir = Path(upstream_dir).resolve(strict=True)
    overlay_dir = Path(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    indexed = _index_git_records(git_files)

    input_names = (
        "chat_template.jinja",
        "genai_config.json",
        "tokenizer_config.json",
    )
    for name in input_names:
        record = indexed.get(name)
        if record is None:
            raise ValueError(f"Git metadata is missing {name}")
        _validate_input_git_record(upstream_dir, record)

    genai_config = json.loads(
        (upstream_dir / "genai_config.json").read_text(encoding="utf-8")
    )
    tokenizer_config = json.loads(
        (upstream_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    chat_template = (upstream_dir / "chat_template.jinja").read_text(
        encoding="utf-8"
    )

    _write_json(
        overlay_dir / "config.json",
        normalize_config(genai_config),
    )
    _write_json(
        overlay_dir / "tokenizer_config.json",
        normalize_tokenizer_config(
            tokenizer_config,
            chat_template,
            genai_config,
        ),
    )
    manifest_tool.generate_manifest(
        upstream_dir,
        overlay_dir / "corelib_phi4_manifest.json",
        True,
        file_metadata=_manifest_file_metadata(indexed),
    )

    generated_files = [
        "config.json",
        "corelib_phi4_manifest.json",
        "tokenizer_config.json",
    ]
    provenance = build_provenance(
        upstream_dir,
        overlay_dir,
        upstream_commit,
        generated_files,
        git_files,
    )
    _write_json(overlay_dir / "provenance.json", provenance)
    return provenance


def build_catalog_entry(
    overlay_dir: Path,
    git_files: list[dict[str, object]],
) -> dict[str, object]:
    overlay_dir = Path(overlay_dir)
    overlay_names = (
        "config.json",
        "corelib_phi4_manifest.json",
        "tokenizer_config.json",
    )
    overlays = {
        name: {
            "path": f"{overlay_dir.name}/{name}",
            **_sha256_record(overlay_dir / name),
        }
        for name in overlay_names
    }

    indexed = _index_git_records(git_files)
    final_files = sorted(set(indexed) | set(overlays))
    remote_size = sum(
        int(record["size"])
        for path, record in indexed.items()
        if path not in overlays
    )
    overlay_size = sum(int(record["size"]) for record in overlays.values())
    size = remote_size + overlay_size
    footprint = round(size / (1024**3), 2)
    return {
        "name": "Phi-4-mini-instruct-oga-dml-AIE4",
        "url": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_COMMIT,
        "file_url": UPSTREAM_API_URL,
        "size": size,
        "default_context_length": 4096,
        "max_prefill_len": 4096,
        "details": {
            "family": "phi4",
            "think": False,
            "think_toggleable": False,
            "parameter_size": "4B",
            "quantization_level": "MatMulNBits Q4",
            "execution_backend": "corelib_aie4",
        },
        "flm_min_version": FLM_MIN_VERSION,
        "vlm": False,
        "modelscope_supported": False,
        "files": final_files,
        "bundled_overlays": overlays,
        "footprint": footprint,
    }


def updated_catalog_documents(
    model_list: dict[str, object],
    model_info: dict[str, object],
    entry: dict[str, object],
    git_files: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    updated_list = copy.deepcopy(model_list)
    models = _require_mapping(updated_list.get("models"), "models")
    reordered: dict[str, object] = {}
    inserted = False
    for name, value in models.items():
        reordered[name] = value
        if name == "phi4-mini-it":
            reordered["phi4-mini-it-aie4"] = {"4b": copy.deepcopy(entry)}
            inserted = True
    if not inserted:
        reordered["phi4-mini-it-aie4"] = {"4b": copy.deepcopy(entry)}
    updated_list["models"] = reordered

    updated_info = copy.deepcopy(model_info)
    updated_info["phi4-mini-it-aie4:4b"] = sorted(
        copy.deepcopy(git_files),
        key=lambda record: record["path"],
    )
    return updated_list, updated_info


_LFS_POINTER = re.compile(
    rb"version https://git-lfs\.github\.com/spec/v1\r?\n"
    rb"oid sha256:([0-9a-fA-F]{64})\r?\n"
    rb"size ([0-9]+)\r?\n?"
)


def git_metadata_records(
    git_dir: Path,
    commit: str,
) -> list[dict[str, object]]:
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"upstream commit must be pinned to {UPSTREAM_COMMIT}")
    repository = str(Path(git_dir))
    resolved = subprocess.check_output(
        ["git", "-C", repository, "rev-parse", commit],
        text=True,
    ).strip()
    if resolved != commit:
        raise ValueError(
            f"metadata checkout resolved {resolved}, expected {commit}"
        )
    tree = subprocess.check_output(
        ["git", "-C", repository, "ls-tree", "-r", "--long", commit],
        text=True,
    )
    records: list[dict[str, object]] = []
    line_pattern = re.compile(
        r"^[0-9]+ blob ([0-9a-f]{40})\s+([0-9]+)\t(.+)$"
    )
    for line in tree.splitlines():
        match = line_pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"unexpected git ls-tree record: {line}")
        oid, pointer_size_text, path = match.groups()
        pointer_size = int(pointer_size_text)
        content = subprocess.check_output(
            ["git", "-C", repository, "show", f"{commit}:{path}"]
        )
        lfs_match = _LFS_POINTER.fullmatch(content)
        record: dict[str, object] = {
            "type": "file",
            "oid": oid,
            "size": pointer_size,
            "path": path,
        }
        if lfs_match is not None:
            logical_size = int(lfs_match.group(2))
            record["size"] = logical_size
            record["lfs"] = {
                "oid": lfs_match.group(1).decode("ascii").lower(),
                "size": logical_size,
                "pointerSize": pointer_size,
            }
        records.append(record)
    return sorted(records, key=lambda record: record["path"])


def huggingface_metadata_records(
    git_dir: Path,
    commit: str,
) -> list[dict[str, object]]:
    request = urllib.request.Request(
        UPSTREAM_API_URL,
        headers={"User-Agent": "FastFlowLM-model-packager/1"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code not in {401, 403}:
            raise
        return git_metadata_records(git_dir, commit)

    if not isinstance(payload, list):
        raise ValueError("Hugging Face tree response must be an array")
    records = [
        record
        for record in payload
        if isinstance(record, dict) and record.get("type") == "file"
    ]
    if len(records) != len(payload):
        raise ValueError("Hugging Face recursive tree contains non-file records")
    normalized: list[dict[str, object]] = []
    for record in records:
        value: dict[str, object] = {
            "type": "file",
            "oid": record["oid"],
            "size": record["size"],
            "path": record["path"],
        }
        if "lfs" in record:
            lfs = _require_mapping(record["lfs"], f"{record['path']}.lfs")
            value["lfs"] = {
                "oid": lfs["oid"],
                "size": lfs["size"],
                "pointerSize": lfs["pointerSize"],
            }
        normalized.append(value)
    return sorted(normalized, key=lambda record: record["path"])


def update_catalog_files(
    model_list_path: Path,
    model_info_path: Path,
    overlay_dir: Path,
    git_files: list[dict[str, object]],
) -> None:
    model_list = json.loads(
        Path(model_list_path).read_text(encoding="utf-8")
    )
    model_info = json.loads(
        Path(model_info_path).read_text(encoding="utf-8")
    )
    entry = build_catalog_entry(overlay_dir, git_files)
    updated_list, updated_info = updated_catalog_documents(
        model_list,
        model_info,
        entry,
        git_files,
    )
    _write_json(Path(model_list_path), updated_list)
    _write_json(Path(model_info_path), updated_info)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    overlay = subparsers.add_parser("generate-overlay")
    overlay.add_argument("--upstream-dir", type=Path, required=True)
    overlay.add_argument("--git-dir", type=Path, required=True)
    overlay.add_argument("--output-dir", type=Path, required=True)
    overlay.add_argument(
        "--upstream-commit",
        default=UPSTREAM_COMMIT,
        choices=[UPSTREAM_COMMIT],
    )

    catalog = subparsers.add_parser("refresh-catalog")
    catalog.add_argument("--git-dir", type=Path, required=True)
    catalog.add_argument("--overlay-dir", type=Path, required=True)
    catalog.add_argument("--model-list", type=Path, required=True)
    catalog.add_argument("--model-info", type=Path, required=True)
    catalog.add_argument(
        "--upstream-commit",
        default=UPSTREAM_COMMIT,
        choices=[UPSTREAM_COMMIT],
    )

    args = parser.parse_args()
    records = huggingface_metadata_records(
        args.git_dir,
        args.upstream_commit,
    )
    if args.command == "generate-overlay":
        generate_overlay(
            args.upstream_dir,
            args.output_dir,
            args.upstream_commit,
            records,
        )
    else:
        update_catalog_files(
            args.model_list,
            args.model_info,
            args.overlay_dir,
            records,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
