from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import onnx
from onnx import TensorProto


SCHEMA_VERSION = 1
MAX_U64 = (1 << 64) - 1
EMBEDDED_INITIALIZERS_FILE = "corelib_embedded_initializers.bin"

MODEL_IDENTITY: dict[str, object] = {
    "family": "phi4",
    "layers": 32,
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_heads": 24,
    "kv_heads": 8,
    "head_size": 128,
    "vocab_size": 200064,
    "group_size": 128,
    "rope_dim": 96,
    "rms_epsilon": 0.00001,
}

_DTYPE_INFO = {
    TensorProto.UINT8: ("uint8", 1),
    TensorProto.FLOAT16: ("float16", 2),
    TensorProto.FLOAT: ("float32", 4),
    TensorProto.INT64: ("int64", 8),
}


def _positive_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def expected_matmul_components(
    k: int,
    n: int,
    group: int,
) -> dict[str, tuple[object, list[int]]]:
    """Return the exact ONNX MatMulNBits component contracts."""
    k = _positive_integer(k, "k")
    n = _positive_integer(n, "n")
    group = _positive_integer(group, "group")
    if k % 2 != 0:
        raise ValueError("k must be even for nibble-packed qweight")
    if k % group != 0:
        raise ValueError("k must be divisible by group")

    groups = k // group
    return {
        "qweight": ("uint8", [n, k // 2]),
        "scales": ({"float16", "float32"}, [n, groups]),
        "qzeros": ("uint8", [n, (groups + 1) // 2]),
    }


def _component_role(
    role: str,
    weight_object: str,
    dtype: object,
    shape: list[int],
) -> dict[str, object]:
    dtypes = {dtype} if isinstance(dtype, str) else set(dtype)
    return {
        "role": role,
        "weight_object": weight_object,
        "dtypes": dtypes,
        "shape": list(shape),
    }


def matmul_roles(
    prefix: str,
    k: int,
    n: int,
    group: int,
) -> dict[str, dict[str, object]]:
    """Return fully qualified initializer contracts for one MatMul object."""
    if not prefix:
        raise ValueError("MatMul initializer prefix must not be empty")
    expected = expected_matmul_components(k, n, group)
    return {
        f"{prefix}.{component}": _component_role(
            f"matmul.{component}",
            prefix,
            dtype,
            shape,
        )
        for component, (dtype, shape) in expected.items()
    }


def ssmlp_roles(layer: int) -> dict[str, dict[str, object]]:
    """Return exact projection and norm contracts for one fused SSMLP."""
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("layer must be an integer")
    if layer < 0 or layer >= 32:
        raise ValueError("layer must be in [0, 31]")

    base = f"model.layers.{layer}"
    weight_object = f"{base}.ssmlp"
    roles: dict[str, dict[str, object]] = {}
    for projection, k, n in (
        ("gate", 3072, 8192),
        ("up", 3072, 8192),
        ("down", 8192, 3072),
    ):
        prefix = f"{base}.mlp.{projection}_proj.MatMulNBits"
        for component, (dtype, shape) in expected_matmul_components(
            k,
            n,
            128,
        ).items():
            roles[f"{prefix}.{component}"] = _component_role(
                f"ssmlp.{projection}.{component}",
                weight_object,
                dtype,
                shape,
            )

    norm_dtypes = {"float16", "float32"}
    roles[f"{base}.post_attention_layernorm.weight"] = {
        "role": "ssmlp.norm0",
        "weight_object": weight_object,
        "dtypes": set(norm_dtypes),
        "shape": [3072],
    }
    next_norm = (
        "model.layers.32.final_norm_layernorm.weight"
        if layer == 31
        else f"model.layers.{layer + 1}.input_layernorm.weight"
    )
    roles[next_norm] = {
        "role": "ssmlp.norm1",
        "weight_object": weight_object,
        "dtypes": set(norm_dtypes),
        "shape": [3072],
    }
    return roles


def host_role(role: str) -> dict[str, object]:
    """Return the accepted source contract for one host tensor."""
    if role == "embedding":
        return {
            "role": role,
            "dtypes": {"float16"},
            "shape": [200064, 3072],
        }
    if role == "input_norm":
        return {
            "role": role,
            "dtypes": {"float16", "float32"},
            "shape": [3072],
        }
    if role in {"cos_cache", "sin_cache"}:
        return {
            "role": role,
            "dtypes": {"float16", "float32"},
            "rank": 2,
            "minimum_shape": [4096, 48],
        }
    raise ValueError(f"unknown host tensor role: {role}")


def _merge_roles(
    destination: dict[str, dict[str, object]],
    additions: dict[str, dict[str, object]],
) -> None:
    duplicates = sorted(destination.keys() & additions.keys())
    if duplicates:
        raise RuntimeError(
            "initializer role generated more than once: " + duplicates[0]
        )
    destination.update(additions)


def required_initializer_roles() -> dict[str, dict[str, object]]:
    """Return all 743 required source initializers in the driver name map."""
    roles: dict[str, dict[str, object]] = {}
    for layer in range(32):
        base = f"model.layers.{layer}"
        # The driver maps o_proj from Q_DIM to HIDDEN. Both are 3072 for
        # Phi-4-mini, but keeping both logical dimensions explicit prevents a
        # future name-only inference from changing the source contract.
        for projection, k, n in (
            ("attn.q_proj", 3072, 3072),
            ("attn.k_proj", 3072, 1024),
            ("attn.v_proj", 3072, 1024),
            ("attn.o_proj", 3072, 3072),
        ):
            prefix = f"{base}.{projection}.MatMulNBits"
            _merge_roles(roles, matmul_roles(prefix, k, n, 128))
        _merge_roles(roles, ssmlp_roles(layer))

    _merge_roles(
        roles,
        matmul_roles("lm_head.MatMulNBits", 3072, 200064, 128),
    )
    _merge_roles(
        roles,
        {
            "model.embed_tokens.weight": host_role("embedding"),
            "model.layers.0.input_layernorm.weight": host_role("input_norm"),
            "cos_cache": host_role("cos_cache"),
            "sin_cache": host_role("sin_cache"),
        },
    )

    weight_objects = {
        record["weight_object"]
        for record in roles.values()
        if "weight_object" in record
    }
    if len(roles) != 743 or len(weight_objects) != 161:
        raise RuntimeError(
            "internal Phi-4 role map mismatch: "
            f"{len(roles)} initializers, {len(weight_objects)} weight objects"
        )
    return roles


def _parse_u64(value: str, field: str, initializer: str) -> int:
    if not value or not value.isdecimal():
        raise ValueError(
            f"{initializer}: external {field} must be an unsigned decimal"
        )
    parsed = int(value)
    if parsed > MAX_U64:
        raise ValueError(f"{initializer}: external {field} exceeds uint64")
    return parsed


def _safe_location(location: str, initializer: str) -> str:
    if not location or "\x00" in location:
        raise ValueError(f"{initializer}: invalid external location")

    windows = PureWindowsPath(location)
    posix = PurePosixPath(location.replace("\\", "/"))
    if (
        windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.is_absolute()
        or bool(posix.root)
    ):
        raise ValueError(
            f"{initializer}: external location must be a relative path"
        )

    parts = tuple(
        part
        for part in posix.parts
        if part not in {"", "."}
    )
    if not parts or ".." in parts:
        raise ValueError(
            f"{initializer}: external location contains path traversal"
        )
    return PurePosixPath(*parts).as_posix()


def _external_source(
    tensor: TensorProto,
    initializer: str,
    model_dir: Path,
) -> tuple[str, Path, int, int]:
    metadata: dict[str, str] = {}
    for item in tensor.external_data:
        if item.key in metadata:
            raise ValueError(
                f"{initializer}: duplicate external metadata key {item.key}"
            )
        metadata[item.key] = item.value

    if "location" not in metadata:
        raise ValueError(f"{initializer}: missing external location")
    if "length" not in metadata:
        raise ValueError(f"{initializer}: missing external length")

    location = _safe_location(metadata["location"], initializer)
    offset = _parse_u64(metadata.get("offset", "0"), "offset", initializer)
    length = _parse_u64(metadata["length"], "length", initializer)
    if length == 0:
        raise ValueError(f"{initializer}: external length must be positive")
    if offset > MAX_U64 - length:
        raise ValueError(f"{initializer}: external range overflow")

    source = model_dir.joinpath(*PurePosixPath(location).parts)
    try:
        resolved = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            f"{initializer}: external file does not exist: {location}"
        ) from error
    if not resolved.is_relative_to(model_dir):
        raise ValueError(
            f"{initializer}: external location escapes the model directory"
        )
    if not resolved.is_file():
        raise ValueError(
            f"{initializer}: external location is not a file: {location}"
        )
    return location, resolved, offset, length


def _dtype_and_item_size(
    tensor: TensorProto,
    initializer: str,
) -> tuple[str, int]:
    try:
        return _DTYPE_INFO[tensor.data_type]
    except KeyError as error:
        type_name = TensorProto.DataType.Name(tensor.data_type)
        raise ValueError(
            f"{initializer}: unsupported ONNX dtype {type_name}"
        ) from error


def _checked_byte_count(
    shape: list[int],
    item_size: int,
    initializer: str,
) -> int:
    elements = 1
    for dimension in shape:
        if dimension <= 0:
            raise ValueError(
                f"{initializer}: shape dimensions must be positive"
            )
        if elements > MAX_U64 // dimension:
            raise ValueError(f"{initializer}: shape element count overflow")
        elements *= dimension
    if elements > MAX_U64 // item_size:
        raise ValueError(f"{initializer}: tensor byte count overflow")
    return elements * item_size


def _validate_contract(
    initializer: str,
    tensor: TensorProto,
    contract: dict[str, object],
) -> tuple[str, int, list[int], int]:
    dtype, item_size = _dtype_and_item_size(tensor, initializer)
    accepted_dtypes = contract.get("dtypes")
    if not isinstance(accepted_dtypes, set) or dtype not in accepted_dtypes:
        expected = ", ".join(sorted(accepted_dtypes or ()))
        raise ValueError(
            f"{initializer}: dtype {dtype} does not match {expected}"
        )

    shape = [int(dimension) for dimension in tensor.dims]
    if "shape" in contract:
        expected_shape = contract["shape"]
        if shape != expected_shape:
            raise ValueError(
                f"{initializer}: shape {shape} does not match "
                f"{expected_shape}"
            )
    else:
        rank = contract.get("rank")
        minimum_shape = contract.get("minimum_shape")
        if len(shape) != rank:
            raise ValueError(
                f"{initializer}: shape rank {len(shape)} does not match {rank}"
            )
        if (
            not isinstance(minimum_shape, list)
            or len(minimum_shape) != len(shape)
            or any(
                actual < minimum
                for actual, minimum in zip(shape, minimum_shape)
            )
        ):
            raise ValueError(
                f"{initializer}: shape {shape} is smaller than "
                f"{minimum_shape}"
            )

    byte_count = _checked_byte_count(shape, item_size, initializer)
    return dtype, item_size, shape, byte_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, full_hash: bool) -> dict[str, object]:
    size = path.stat().st_size
    if size < 0 or size > MAX_U64:
        raise ValueError(f"file size exceeds uint64: {path}")
    record: dict[str, object] = {"size": size}
    if full_hash:
        record["sha256"] = _sha256(path)
    return record


def _load_initializers(model_path: Path) -> dict[str, TensorProto]:
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception as error:
        raise ValueError(f"failed to parse ONNX model {model_path}") from error

    initializers: dict[str, TensorProto] = {}
    for tensor in model.graph.initializer:
        if tensor.name in initializers:
            raise ValueError(f"duplicate initializer: {tensor.name}")
        initializers[tensor.name] = tensor
    return initializers


def _initializer_record(
    contract: dict[str, object],
    *,
    dtype: str,
    shape: list[int],
    file: str,
    offset: int,
    length: int,
) -> dict[str, object]:
    record: dict[str, object] = {
        "file": file,
        "offset": offset,
        "length": length,
        "dtype": dtype,
        "shape": shape,
        "role": contract["role"],
    }
    return record


def _generate_manifest(
    model_dir: Path,
    output: Path,
    full_hash: bool,
    roles: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Generate a manifest using an explicit role map.

    The public generator always supplies ``required_initializer_roles()``.
    The explicit map keeps small synthetic unit models practical without
    allocating the accepted model's multi-gigabyte tensors.
    """
    model_dir = Path(model_dir).resolve(strict=True)
    output = Path(output)
    if not model_dir.is_dir():
        raise ValueError(f"model directory is not a directory: {model_dir}")
    if not isinstance(full_hash, bool):
        raise ValueError("full_hash must be a boolean")

    model_path = model_dir / "model.onnx"
    if not model_path.is_file():
        raise ValueError(f"model.onnx does not exist in {model_dir}")
    if not output.parent.exists() or not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")

    initializers = _load_initializers(model_path)
    missing = sorted(set(roles) - set(initializers))
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += f", ... ({len(missing)} total)"
        raise ValueError(f"missing initializer(s): {preview}")

    records: dict[str, dict[str, object]] = {}
    external_files: dict[str, Path] = {}
    embedded: list[tuple[str, bytes]] = []
    embedded_offset = 0

    for name in sorted(roles):
        tensor = initializers[name]
        contract = roles[name]
        dtype, item_size, shape, byte_count = _validate_contract(
            name,
            tensor,
            contract,
        )
        is_external = (
            tensor.data_location == TensorProto.EXTERNAL
            or bool(tensor.external_data)
        )
        if is_external:
            location, path, offset, length = _external_source(
                tensor,
                name,
                model_dir,
            )
            if length != byte_count:
                raise ValueError(
                    f"{name}: external byte count {length} does not match "
                    f"dtype/shape byte count {byte_count}"
                )
            if offset % item_size != 0:
                raise ValueError(
                    f"{name}: external offset is not dtype-aligned"
                )
            size = path.stat().st_size
            if size > MAX_U64:
                raise ValueError(f"{name}: external file size exceeds uint64")
            if offset > size or length > size - offset:
                raise ValueError(
                    f"{name}: external range exceeds file size"
                )
            previous = external_files.setdefault(location, path)
            if previous != path:
                raise ValueError(
                    f"{name}: external location resolves inconsistently"
                )
            records[name] = _initializer_record(
                contract,
                dtype=dtype,
                shape=shape,
                file=location,
                offset=offset,
                length=length,
            )
            continue

        raw_data = bytes(tensor.raw_data)
        if not raw_data:
            raise ValueError(
                f"{name}: embedded initializer must use raw_data"
            )
        if len(raw_data) != byte_count:
            raise ValueError(
                f"{name}: embedded byte count {len(raw_data)} does not match "
                f"dtype/shape byte count {byte_count}"
            )
        if embedded_offset > MAX_U64 - byte_count:
            raise ValueError("embedded initializer sidecar offset overflow")
        records[name] = _initializer_record(
            contract,
            dtype=dtype,
            shape=shape,
            file=EMBEDDED_INITIALIZERS_FILE,
            offset=embedded_offset,
            length=byte_count,
        )
        embedded.append((name, raw_data))
        embedded_offset += byte_count

    sidecar_path = model_dir / EMBEDDED_INITIALIZERS_FILE
    if embedded and EMBEDDED_INITIALIZERS_FILE in external_files:
        raise ValueError(
            "embedded initializer sidecar conflicts with an external data file"
        )

    protected_paths = {model_path.resolve()}
    protected_paths.update(external_files.values())
    if embedded:
        protected_paths.add(sidecar_path.resolve(strict=False))
    output_resolved = output.resolve(strict=False)
    if output_resolved in protected_paths:
        raise ValueError("output path would overwrite ONNX initializer data")

    if embedded:
        with sidecar_path.open("wb") as stream:
            for _, raw_data in embedded:
                stream.write(raw_data)

    files: dict[str, dict[str, object]] = {
        "model.onnx": _file_record(model_path, full_hash)
    }
    for location in sorted(external_files):
        files[location] = _file_record(external_files[location], full_hash)
    if embedded:
        files[EMBEDDED_INITIALIZERS_FILE] = _file_record(
            sidecar_path,
            full_hash,
        )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": "corelib_aie4",
        "model": dict(MODEL_IDENTITY),
        "backend": {"max_seq": 4096},
        "files": files,
        "initializers": records,
    }
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output.write_text(serialized + "\n", encoding="utf-8", newline="\n")
    return manifest


def generate_manifest(
    model_dir: Path,
    output: Path,
    full_hash: bool,
) -> dict[str, object]:
    return _generate_manifest(
        model_dir,
        output,
        full_hash,
        required_initializer_roles(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-hash", action="store_true")
    args = parser.parse_args()
    generate_manifest(args.model_dir, args.output, args.full_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
