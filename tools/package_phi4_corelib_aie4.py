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

# Design 8.1 / `PACKAGE-1`. The assembled model directory has two provenances
# and they are validated separately, because a rule requiring every catalog
# file to carry Hugging Face metadata cannot succeed: these files are
# FastFlow-authored and do not exist upstream by construction.
#
# `tokenizer_config.json` is the one overlay that also exists upstream. The
# overlay shadows it because the published file carries neither a chat template
# nor eos_token_id, so it is downloaded from nowhere and its upstream size is
# not counted, but it legitimately has an upstream record. The other three must
# never acquire one: an upstream record for `config.json`,
# `corelib_phi4_manifest.json` or `provenance.json` would mean FastFlow's
# contract was published to the model repository, which is a provenance error
# and not a convenience.
OVERLAY_FILES = (
    "config.json",
    "corelib_phi4_manifest.json",
    "provenance.json",
    "tokenizer_config.json",
)
OVERLAY_FILES_WITHOUT_UPSTREAM = (
    "config.json",
    "corelib_phi4_manifest.json",
    "provenance.json",
)

# Upstream files deliberately not carried into the assembled package.
#
# `genai_config.json` is excluded by `MODEL-2`: flm.exe runs no ORT or genai
# graph, and shipping an unused configuration invites a future reader to treat
# it as authoritative. `.gitattributes` is a Git repository artifact rather
# than a model file. `chat_template.jinja` is deliberately NOT excluded: it is
# the verbatim source of the template the overlay inlines, and keeping it is
# what makes the overlay auditable on the target machine.
EXCLUDED_UPSTREAM_FILES = (
    ".gitattributes",
    "genai_config.json",
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


# The two catalog documents are NOT written with `_write_json`, and the reason
# is worth stating because the obvious thing was tried and it was wrong.
#
# `model_list.json` is the product's most actively edited file: 2,234 lines, 27
# model families, and an upstream that adds to it constantly. A `json.dump` of
# the whole document for a single added entry changed indentation (4 -> 2),
# line endings (LF -> CRLF) and key order (alphabetised) across every one of
# those lines. It was semantics-preserving and it was still the wrong output:
# it conflicts with any concurrent model addition, and the one real change is
# invisible in review.
#
# `model_info.json` never shipped that churn only because the entry that was
# committed was not written by this tool. Run `refresh-catalog` against the
# committed file and `_write_json` reformats all 3,387 of its lines the same
# way. Fixing one file and leaving the other would have moved the defect
# rather than closed it, so both go through the writer below.
#
# Matching the style instead of preserving it does not work either. Neither
# committed file is the output of any `json.dumps` call -- `model_list.json`
# carries hand-authored details a serialiser normalises away (`"label":[` with
# no space, `"9b":{`, a trailing-whitespace line, a stray blank line) and
# `model_info.json` has no trailing newline. A re-dump of the catalog at
# indent 4 with the original key order still rewrites 66 lines.
#
# So the entry is spliced in as text and every other byte is left exactly as it
# was found. Textual editing of JSON is easy to get subtly wrong, so the writer
# proves both halves of its own contract before it writes: the produced text
# must parse to the object graph the caller asked for, and it must reduce to
# the input byte-for-byte once the entry is removed again.
CATALOG_MODEL_NAME = "phi4-mini-it-aie4"
CATALOG_MODEL_ANCHOR = "phi4-mini-it"
CATALOG_INFO_KEY = f"{CATALOG_MODEL_NAME}:4b"
_CATALOG_MODELS_KEY = "models"
_CATALOG_MODEL_DEPTH = 2
_CATALOG_INFO_DEPTH = 1


def _json_text_style(text: str) -> tuple[str, str]:
    """The document's newline and its one indent step, read out of itself."""
    newline = "\r\n" if "\r\n" in text else "\n"
    for line in text.split(newline):
        stripped = line.lstrip(" \t")
        if stripped and stripped != line:
            return newline, line[: len(line) - len(stripped)]
    raise ValueError("cannot determine the catalog's indent step")


_JSON_CLOSERS = {"{": "}", "[": "]"}


def _json_value_span(text: str, search_from: int) -> tuple[int, int]:
    """Half-open span of the object or array at or after `search_from`.

    Bracket counting with string awareness rather than a regex: model names
    and URLs contain braces and escaped quotes, and a regex that gets that
    wrong fails by silently truncating somebody else's entry. Arrays count
    because `model_info.json` maps each model to a list of file records.
    """
    candidates = [
        position
        for position in (
            text.find("{", search_from),
            text.find("[", search_from),
        )
        if position >= 0
    ]
    if not candidates:
        raise ValueError("no JSON object or array in the catalog")
    open_at = min(candidates)
    closer = _JSON_CLOSERS[text[open_at]]
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_at, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == text[open_at]:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return open_at, index + 1
    raise ValueError("unterminated JSON value in the catalog")


def _member_span(
    text: str,
    name: str,
    depth: int,
    newline: str,
    indent_unit: str,
) -> tuple[int, int] | None:
    """Span of `"name": <value>` written at `depth` indent steps, or None.

    The indent is part of the match on purpose. `"phi4-mini-it"` also occurs
    inside URLs and inside the nested `4b` object of the entry being written;
    only the occurrence at the members' own level is the member.
    """
    marker = f"{newline}{indent_unit * depth}{json.dumps(name)}:"
    first = text.find(marker)
    if first < 0:
        return None
    if text.find(marker, first + 1) >= 0:
        raise ValueError(f"catalog has more than one {name} member")
    start = first + len(newline)
    _, end = _json_value_span(text, first + len(marker))
    return start, end


def _rendered_member(
    name: str,
    value: object,
    depth: int,
    newline: str,
    indent_unit: str,
) -> str:
    """`"<name>": <value>` in the document's own style."""
    rendered = json.dumps(
        {name: value},
        ensure_ascii=False,
        indent=indent_unit,
    )
    # Strip the wrapper object `json.dumps` needed in order to emit the key,
    # then push the remainder out to the depth it will live at.
    body = rendered.split("\n")[1:-1]
    extra = indent_unit * (depth - 1)
    return newline.join(extra + line for line in body)


def _splice_member(
    text: str,
    name: str,
    value: object,
    depth: int,
    anchor_name: str | None,
    container_name: str | None,
) -> str:
    """Insert or replace one member, touching nothing else.

    `anchor_name` is the member the new one is placed after when it is not
    already present; `container_name` is the object it belongs to, or None for
    the document root. Both mirror `updated_catalog_documents`, so the text and
    the object graph agree about where the entry goes.
    """
    newline, indent_unit = _json_text_style(text)
    member = _rendered_member(name, value, depth, newline, indent_unit)

    existing = _member_span(text, name, depth, newline, indent_unit)
    if existing is not None:
        # Re-running the tool must be idempotent. Replacing in place also keeps
        # the entry where a previous run put it rather than moving it.
        start, end = existing
        return text[:start] + member + text[end:]

    if anchor_name is not None:
        anchor = _member_span(text, anchor_name, depth, newline, indent_unit)
        if anchor is not None:
            # The insert lands between the anchor's closing brace and the
            # comma that already follows it, so the anchor's own line is
            # untouched and the comma ends up closing the new member instead.
            insert_at = anchor[1]
            return text[:insert_at] + "," + newline + member + text[insert_at:]

    if container_name is None:
        container = _json_value_span(text, 0)
    else:
        found = _member_span(
            text, container_name, depth - 1, newline, indent_unit
        )
        if found is None:
            raise ValueError(f"catalog has no {container_name} member")
        container = found
    # No anchor: append as the last member, which is what
    # `updated_catalog_documents` does with the same input.
    insert_at = len(text[: container[1] - 1].rstrip())
    return text[:insert_at] + "," + newline + member + text[insert_at:]


def _strip_member(text: str, name: str, depth: int) -> str:
    """`text` with that member and its separator comma removed."""
    newline, indent_unit = _json_text_style(text)
    span = _member_span(text, name, depth, newline, indent_unit)
    if span is None:
        return text
    start, end = span
    head = text[:start]
    suffix = indent_unit * depth
    if head.endswith(suffix):
        head = head[: -len(suffix)]
    if head.endswith(newline):
        head = head[: -len(newline)]
    tail = text[end:]
    if head.endswith(","):
        head = head[:-1]
    elif tail.startswith(","):
        tail = tail[1:]
    return head + tail


def _write_spliced(
    path: Path,
    original: str,
    updated: object,
    value: object,
    name: str,
    depth: int,
    anchor_name: str | None,
    container_name: str | None,
) -> None:
    """Splice one member in and write, but only if the result checks out.

    Two checks, both before anything reaches disk, because a textual edit that
    is subtly wrong is worse than a reformat: a reformat is merely noisy.
    """
    produced = _splice_member(
        original, name, value, depth, anchor_name, container_name
    )
    if json.loads(produced) != updated:
        raise ValueError(
            f"the spliced {name} member does not parse to the requested "
            "document"
        )
    if _strip_member(produced, name, depth) != _strip_member(
        original, name, depth
    ):
        raise ValueError(
            f"the spliced document changed bytes outside the {name} entry"
        )
    Path(path).write_bytes(produced.encode("utf-8"))


def write_model_list(path: Path, original: str, updated_list: object) -> None:
    """Write `model_list.json` with only the AIE4 entry changed."""
    value = _require_mapping(
        _require_mapping(updated_list, "catalog").get(_CATALOG_MODELS_KEY),
        _CATALOG_MODELS_KEY,
    ).get(CATALOG_MODEL_NAME)
    if value is None:
        raise ValueError(f"updated catalog has no {CATALOG_MODEL_NAME} entry")
    _write_spliced(
        path,
        original,
        updated_list,
        value,
        CATALOG_MODEL_NAME,
        _CATALOG_MODEL_DEPTH,
        CATALOG_MODEL_ANCHOR,
        _CATALOG_MODELS_KEY,
    )


def write_model_info(path: Path, original: str, updated_info: object) -> None:
    """Write `model_info.json` with only the AIE4 record list changed.

    No anchor: `updated_catalog_documents` appends this key, and the committed
    file already has it last. Preserving that keeps the file a fixed point.
    """
    value = _require_mapping(updated_info, "metadata").get(CATALOG_INFO_KEY)
    if value is None:
        raise ValueError(f"updated metadata has no {CATALOG_INFO_KEY} entry")
    _write_spliced(
        path,
        original,
        updated_info,
        value,
        CATALOG_INFO_KEY,
        _CATALOG_INFO_DEPTH,
        None,
        None,
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
    # Read bytes and decode explicitly. `read_text` applies universal newline
    # translation, which would turn a CRLF template into an LF string and break
    # the byte-equality that `_require_inlined_template_matches_upstream`
    # depends on.
    chat_template_bytes = (upstream_dir / "chat_template.jinja").read_bytes()
    chat_template = chat_template_bytes.decode("utf-8")

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
    _require_inlined_template_matches_upstream(overlay_dir)
    return provenance


def build_catalog_entry(
    overlay_dir: Path,
    git_files: list[dict[str, object]],
) -> dict[str, object]:
    overlay_dir = Path(overlay_dir)
    overlays = {
        name: {
            "path": f"{overlay_dir.name}/{name}",
            **_sha256_record(overlay_dir / name),
        }
        for name in OVERLAY_FILES
    }

    indexed = _index_git_records(git_files)
    upstream_files = {
        path
        for path in indexed
        if path not in overlays and path not in EXCLUDED_UPSTREAM_FILES
    }
    final_files = sorted(upstream_files | set(overlays))
    # `size` and `footprint` cover the assembled on-disk directory: the
    # upstream files actually downloaded plus the overlay files shipped inside
    # FastFlow. They are not the upstream repository's size, which is larger,
    # and not the overlay's, which is negligible.
    remote_size = sum(int(indexed[path]["size"]) for path in upstream_files)
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


def _require_inlined_template_matches_upstream(overlay_dir: Path) -> None:
    """Require the inlined chat template to equal the upstream jinja file.

    `AutoModel::setup_tokenizer` prefers a standalone `chat_template.jinja`
    over the `chat_template` key in `tokenizer_config.json`: when the file is
    present it *overwrites* the key. Both are in the package, so the upstream
    file wins at run time and the overlay's inlined copy is dead code unless
    the two are byte-identical.

    That makes any drift silent and one-directional -- the overlay would look
    edited while the model kept using the upstream template -- so the equality
    is asserted rather than assumed. The check is offline: `provenance.json`
    records the upstream file's SHA-256, so no download is needed.
    """
    overlay_dir = Path(overlay_dir)
    tokenizer_config = json.loads(
        (overlay_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (overlay_dir / "provenance.json").read_text(encoding="utf-8")
    )
    inlined = tokenizer_config.get("chat_template")
    if not isinstance(inlined, str) or not inlined:
        raise ValueError(
            "overlay tokenizer_config.json has no string chat_template"
        )
    record = _require_mapping(
        _require_mapping(
            _require_mapping(provenance.get("upstream"), "upstream").get(
                "inputs"
            ),
            "upstream.inputs",
        ).get("chat_template.jinja"),
        "upstream.inputs['chat_template.jinja']",
    )
    encoded = inlined.encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != record.get("sha256") or len(encoded) != record.get("size"):
        raise ValueError(
            "the overlay's inlined chat_template does not match the upstream "
            "chat_template.jinja it was generated from. AutoModel prefers the "
            "standalone .jinja file, so the inlined copy would be silently "
            "ignored: regenerate the overlay instead of editing it."
        )


def validate_catalog_provenance(
    entry: dict[str, object],
    upstream_records: list[dict[str, object]],
    overlay_dir: Path,
) -> None:
    """Check the two provenances of the assembled package separately.

    Design `PACKAGE-1`. Upstream files must each carry a Hugging Face metadata
    record at the pinned revision and are what the downloader fetches and
    hash-checks. Overlay files must exist in the shipped overlay directory and,
    unless they shadow a published file, must have no upstream record at all.
    """
    overlay_dir = Path(overlay_dir)
    indexed = _index_git_records(upstream_records)

    if entry.get("revision") != UPSTREAM_COMMIT:
        raise ValueError(
            f"catalog revision must be pinned to {UPSTREAM_COMMIT}"
        )
    if UPSTREAM_COMMIT not in str(entry.get("file_url", "")):
        raise ValueError("catalog file_url must reference the pinned revision")
    if entry.get("flm_min_version") != FLM_MIN_VERSION:
        raise ValueError(
            f"catalog flm_min_version must be {FLM_MIN_VERSION}"
        )
    if entry.get("modelscope_supported") is not False:
        raise ValueError("this tag has no ModelScope publication")
    if "ms_url" in entry:
        raise ValueError("a tag without a ModelScope publication has no ms_url")

    _require_inlined_template_matches_upstream(overlay_dir)

    overlays = _require_mapping(entry.get("bundled_overlays"), "bundled_overlays")
    if set(overlays) != set(OVERLAY_FILES):
        raise ValueError(
            "bundled_overlays must be exactly "
            f"{sorted(OVERLAY_FILES)}, got {sorted(overlays)}"
        )

    catalog_files = entry.get("files")
    if not isinstance(catalog_files, list):
        raise ValueError("catalog files must be a list")
    if sorted(catalog_files) != list(catalog_files):
        raise ValueError("catalog files must be sorted")
    if len(set(catalog_files)) != len(catalog_files):
        raise ValueError("catalog files contains duplicates")

    for name, record in overlays.items():
        if name not in catalog_files:
            raise ValueError(f"overlay {name} is missing from catalog files")
        path = overlay_dir.parent / str(record["path"])
        if not path.is_file():
            raise ValueError(f"overlay file is not installed: {path}")
        actual = _sha256_record(path)
        if actual != {"size": record["size"], "sha256": record["sha256"]}:
            raise ValueError(f"overlay {name} does not match its catalog record")
        if name in OVERLAY_FILES_WITHOUT_UPSTREAM and name in indexed:
            raise ValueError(
                f"overlay {name} has an upstream metadata record. FastFlow's "
                "own package contract appears to have been published to "
                f"{UPSTREAM_REPOSITORY}, which makes the two provenances "
                "indistinguishable."
            )

    for name in catalog_files:
        if name in overlays:
            continue
        if name not in indexed:
            raise ValueError(
                f"upstream file {name} has no Hugging Face metadata record"
            )
        if name in EXCLUDED_UPSTREAM_FILES:
            raise ValueError(
                f"{name} is excluded from the package but is still listed"
            )

    for name in indexed:
        if name in catalog_files or name in EXCLUDED_UPSTREAM_FILES:
            continue
        raise ValueError(
            f"upstream file {name} is neither packaged nor explicitly excluded"
        )

    expected_size = sum(
        int(indexed[name]["size"])
        for name in catalog_files
        if name not in overlays
    ) + sum(int(record["size"]) for record in overlays.values())
    if entry.get("size") != expected_size:
        raise ValueError(
            f"catalog size must be {expected_size}, got {entry.get('size')}"
        )
    expected_footprint = round(expected_size / (1024**3), 2)
    if entry.get("footprint") != expected_footprint:
        raise ValueError(
            f"catalog footprint must be {expected_footprint}, "
            f"got {entry.get('footprint')}"
        )


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
        if name == CATALOG_MODEL_NAME:
            # A re-run reads back its own output. Skipping the stale copy here
            # is what makes the freshly built entry survive to the end of the
            # loop instead of being overwritten by the one on disk.
            continue
        reordered[name] = value
        if name == CATALOG_MODEL_ANCHOR:
            reordered[CATALOG_MODEL_NAME] = {"4b": copy.deepcopy(entry)}
            inserted = True
    if not inserted:
        reordered[CATALOG_MODEL_NAME] = {"4b": copy.deepcopy(entry)}
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
    # Read bytes and decode. `read_text` applies universal newline translation,
    # which would hide the file's real line endings from the style-preserving
    # writer and make it rewrite every line of a CRLF catalog.
    model_list_text = Path(model_list_path).read_bytes().decode("utf-8")
    model_list = json.loads(model_list_text)
    model_info_text = Path(model_info_path).read_bytes().decode("utf-8")
    model_info = json.loads(model_info_text)
    entry = build_catalog_entry(overlay_dir, git_files)
    validate_catalog_provenance(entry, git_files, Path(overlay_dir))
    updated_list, updated_info = updated_catalog_documents(
        model_list,
        model_info,
        entry,
        git_files,
    )
    write_model_list(Path(model_list_path), model_list_text, updated_list)
    write_model_info(Path(model_info_path), model_info_text, updated_info)


def validate_catalog_files(
    model_list_path: Path,
    model_info_path: Path,
    overlay_dir: Path,
) -> None:
    """Re-check the committed catalog without contacting the network.

    Regeneration is not always possible on a machine without a metadata
    checkout, but the committed documents can still be held to the same
    contract, which is what keeps a hand edit from slipping through.
    """
    model_list = json.loads(
        Path(model_list_path).read_text(encoding="utf-8")
    )
    model_info = json.loads(
        Path(model_info_path).read_text(encoding="utf-8")
    )
    entry = model_list["models"]["phi4-mini-it-aie4"]["4b"]
    validate_catalog_provenance(
        entry,
        model_info["phi4-mini-it-aie4:4b"],
        Path(overlay_dir),
    )


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

    validate = subparsers.add_parser("validate-catalog")
    validate.add_argument("--overlay-dir", type=Path, required=True)
    validate.add_argument("--model-list", type=Path, required=True)
    validate.add_argument("--model-info", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "validate-catalog":
        validate_catalog_files(
            args.model_list,
            args.model_info,
            args.overlay_dir,
        )
        return 0

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
