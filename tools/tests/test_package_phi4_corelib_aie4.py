from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper

from tools import generate_phi4_corelib_manifest as manifest_tool
from tools import package_phi4_corelib_aie4 as package_tool


class Phi4CorelibOverlayTests(unittest.TestCase):
    def _genai_config(self) -> dict[str, object]:
        return {
            "model": {
                "bos_token_id": 199999,
                "context_length": 131072,
                "decoder": {
                    "head_size": 128,
                    "hidden_size": 3072,
                    "num_attention_heads": 24,
                    "num_hidden_layers": 32,
                    "num_key_value_heads": 8,
                },
                "eos_token_id": [200020, 199999],
                "type": "phi3",
                "vocab_size": 200064,
            },
            "search": {"max_length": 131072},
        }

    def _git_blob_oid(self, data: bytes) -> str:
        header = f"blob {len(data)}\0".encode("ascii")
        return hashlib.sha1(header + data).hexdigest()

    def _write_accepted_upstream(
        self,
        root: Path,
    ) -> list[dict[str, object]]:
        genai_data = (
            json.dumps(self._genai_config(), indent=2, sort_keys=True) + "\n"
        ).encode()
        tokenizer_data = b'{"model_max_length":131072}\n'
        template_data = b"{{ messages | length }}\n"
        for name, data in (
            ("genai_config.json", genai_data),
            ("tokenizer_config.json", tokenizer_data),
            ("chat_template.jinja", template_data),
        ):
            (root / name).write_bytes(data)

        item_sizes = {
            "uint8": (TensorProto.UINT8, 1),
            "float16": (TensorProto.FLOAT16, 2),
            "float32": (TensorProto.FLOAT, 4),
            "int64": (TensorProto.INT64, 8),
        }
        tensors: list[TensorProto] = []
        offset = 0
        for name, contract in sorted(
            manifest_tool.required_initializer_roles().items()
        ):
            accepted = sorted(contract["dtypes"])
            dtype = "float16" if "float16" in accepted else accepted[0]
            data_type, item_size = item_sizes[dtype]
            shape = list(
                contract.get("shape", contract.get("minimum_shape"))
            )
            if name.endswith(".qweight"):
                shape = [shape[0], shape[1] // 64, 64]
            elif (
                name.endswith(".scales") or
                name.endswith(".qzeros")
            ):
                shape = [shape[0] * shape[1]]
            length = item_size
            for dimension in shape:
                length *= dimension
            tensor = TensorProto()
            tensor.name = name
            tensor.data_type = data_type
            tensor.dims.extend(shape)
            tensor.data_location = TensorProto.EXTERNAL
            for key, value in (
                ("location", "model.onnx.data"),
                ("offset", str(offset)),
                ("length", str(length)),
            ):
                item = tensor.external_data.add()
                item.key = key
                item.value = value
            tensors.append(tensor)
            offset += length

        logical_sha = "b" * 64
        (root / "model.onnx.data").write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{logical_sha}\n"
            f"size {offset}\n",
            encoding="ascii",
            newline="\n",
        )
        model = helper.make_model(
            helper.make_graph([], "accepted-phi4", [], [], tensors)
        )
        model_data = model.SerializeToString()
        (root / "model.onnx").write_bytes(model_data)

        records: list[dict[str, object]] = []
        for name, data in (
            ("chat_template.jinja", template_data),
            ("genai_config.json", genai_data),
            ("tokenizer_config.json", tokenizer_data),
        ):
            records.append(
                {
                    "type": "file",
                    "oid": self._git_blob_oid(data),
                    "size": len(data),
                    "path": name,
                }
            )
        records.extend(
            [
                {
                    "type": "file",
                    "oid": "1" * 40,
                    "size": len(model_data),
                    "lfs": {
                        "oid": hashlib.sha256(model_data).hexdigest(),
                        "size": len(model_data),
                        "pointerSize": 131,
                    },
                    "path": "model.onnx",
                },
                {
                    "type": "file",
                    "oid": "2" * 40,
                    "size": offset,
                    "lfs": {
                        "oid": logical_sha,
                        "size": offset,
                        "pointerSize": 135,
                    },
                    "path": "model.onnx.data",
                },
            ]
        )
        return records

    def test_normalized_config_is_derived_from_oga_and_backend_contract(self):
        self.assertEqual(
            package_tool.normalize_config(self._genai_config()),
            {
                "flm_version": "1.0.4",
                "head_dim": 128,
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "model_type": "phi4",
                "num_attention_heads": 24,
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1.0e-5,
                "vocab_size": 200064,
            },
        )

    def test_normalized_tokenizer_preserves_upstream_and_adds_exact_sources(self):
        upstream = {
            "add_bos_token": False,
            "model_max_length": 131072,
            "tokenizer_class": "GPT2Tokenizer",
        }
        template = "{{ messages | length }}"

        normalized = package_tool.normalize_tokenizer_config(
            upstream,
            template,
            self._genai_config(),
        )

        self.assertEqual(normalized["add_bos_token"], False)
        self.assertEqual(normalized["model_max_length"], 131072)
        self.assertEqual(normalized["tokenizer_class"], "GPT2Tokenizer")
        self.assertEqual(normalized["chat_template"], template)
        self.assertEqual(normalized["eos_token_id"], [200020, 199999])

    def test_normalization_rejects_unapproved_oga_identity(self):
        bad = self._genai_config()
        bad["model"]["decoder"]["hidden_size"] = 4096
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            package_tool.normalize_config(bad)

        bad = self._genai_config()
        bad["model"]["eos_token_id"] = [199999]
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            package_tool.normalize_tokenizer_config({}, "template", bad)

    def test_manifest_accepts_verified_lfs_pointer_as_logical_external_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logical_sha = "a" * 64
            logical_size = 4
            (root / "weights.bin").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{logical_sha}\n"
                f"size {logical_size}\n",
                encoding="ascii",
                newline="\n",
            )

            tensor = TensorProto()
            tensor.name = "test.weight"
            tensor.data_type = TensorProto.UINT8
            tensor.dims.extend([2, 2])
            tensor.data_location = TensorProto.EXTERNAL
            for key, value in (
                ("location", "weights.bin"),
                ("offset", "0"),
                ("length", "4"),
            ):
                item = tensor.external_data.add()
                item.key = key
                item.value = value
            model = helper.make_model(
                helper.make_graph([], "logical-lfs", [], [], [tensor])
            )
            model_path = root / "model.onnx"
            model_path.write_bytes(model.SerializeToString())

            manifest = manifest_tool._generate_manifest(
                root,
                root / "manifest.json",
                True,
                {
                    "test.weight": {
                        "role": "test.tensor",
                        "dtypes": {"uint8"},
                        "shape": [2, 2],
                    }
                },
                file_metadata={
                    "weights.bin": {
                        "size": logical_size,
                        "sha256": logical_sha,
                    }
                },
            )

            self.assertEqual(
                manifest["files"]["weights.bin"],
                {"size": logical_size, "sha256": logical_sha},
            )
            self.assertEqual(
                manifest["files"]["model.onnx"],
                {
                    "size": model_path.stat().st_size,
                    "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                },
            )

    def test_catalog_measurements_use_remote_logical_and_overlay_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "remote-lfs.bin").write_bytes(b"pointer")
            (root / "config.json").write_bytes(b"{}")
            size, footprint = package_tool.catalog_measurements(
                root,
                {"remote-lfs.bin": 1024**3},
            )
            self.assertEqual(size, 1024**3 + 2)
            self.assertEqual(footprint, 1.0)

    def test_provenance_records_pinned_inputs_and_generated_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            overlay = root / "overlay"
            upstream.mkdir()
            overlay.mkdir()
            (upstream / "genai_config.json").write_bytes(b'{"source":1}\n')
            (upstream / "tokenizer_config.json").write_bytes(b'{"source":2}\n')
            (upstream / "chat_template.jinja").write_bytes(b"template\n")
            (overlay / "config.json").write_bytes(b'{"output":1}\n')

            provenance = package_tool.build_provenance(
                upstream,
                overlay,
                "e751fb68c2cfffe6b0d32942118f75ac0a0365bb",
                ["config.json"],
            )

            self.assertEqual(
                provenance["upstream"]["commit"],
                "e751fb68c2cfffe6b0d32942118f75ac0a0365bb",
            )
            self.assertEqual(
                set(provenance["upstream"]["inputs"]),
                {
                    "chat_template.jinja",
                    "genai_config.json",
                    "tokenizer_config.json",
                },
            )
            self.assertEqual(
                provenance["generated"]["config.json"]["size"],
                len(b'{"output":1}\n'),
            )
            self.assertEqual(
                len(provenance["generated"]["config.json"]["sha256"]),
                64,
            )

    def test_overlay_generation_is_complete_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / "upstream"
            first = root / "first"
            second = root / "second"
            upstream.mkdir()
            records = self._write_accepted_upstream(upstream)

            package_tool.generate_overlay(
                upstream,
                first,
                package_tool.UPSTREAM_COMMIT,
                records,
            )
            package_tool.generate_overlay(
                upstream,
                second,
                package_tool.UPSTREAM_COMMIT,
                records,
            )

            expected = {
                "config.json",
                "corelib_phi4_manifest.json",
                "provenance.json",
                "tokenizer_config.json",
            }
            self.assertEqual(
                {path.name for path in first.iterdir()},
                expected,
            )
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in first.iterdir()
                },
                {
                    path.name: path.read_bytes()
                    for path in second.iterdir()
                },
            )
            manifest = json.loads(
                (first / "corelib_phi4_manifest.json").read_text()
            )
            self.assertEqual(len(manifest["initializers"]), 743)
            self.assertEqual(len(manifest["weight_objects"]), 161)
            self.assertEqual(
                set(manifest["files"]),
                {"model.onnx", "model.onnx.data"},
            )
            self.assertNotIn(
                "corelib_embedded_initializers.bin",
                manifest["files"],
            )
            provenance = json.loads(
                (first / "provenance.json").read_text()
            )
            self.assertEqual(
                provenance["upstream"]["commit"],
                package_tool.UPSTREAM_COMMIT,
            )
            self.assertEqual(
                provenance["upstream"]["git_files"],
                sorted(records, key=lambda record: record["path"]),
            )

    def test_catalog_entry_uses_exact_remote_and_overlay_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "phi4-mini-it-aie4"
            overlay.mkdir()
            for name, data in (
                ("config.json", b"config"),
                ("corelib_phi4_manifest.json", b"manifest"),
                ("tokenizer_config.json", b"normalized-tokenizer"),
            ):
                (overlay / name).write_bytes(data)
            records = [
                {
                    "type": "file",
                    "oid": "1" * 40,
                    "size": 100,
                    "path": "model.onnx",
                },
                {
                    "type": "file",
                    "oid": "2" * 40,
                    "size": 200,
                    "path": "tokenizer_config.json",
                },
            ]

            entry = package_tool.build_catalog_entry(overlay, records)

            expected_size = (
                100 + len(b"config") + len(b"manifest") +
                len(b"normalized-tokenizer")
            )
            self.assertEqual(entry["size"], expected_size)
            self.assertEqual(
                entry["footprint"],
                round(expected_size / (1024**3), 2),
            )
            self.assertEqual(
                set(entry["files"]),
                {
                    "config.json",
                    "corelib_phi4_manifest.json",
                    "model.onnx",
                    "tokenizer_config.json",
                },
            )
            self.assertEqual(
                set(entry["bundled_overlays"]),
                {
                    "config.json",
                    "corelib_phi4_manifest.json",
                    "tokenizer_config.json",
                },
            )

    def test_metadata_refresh_preserves_unrelated_entries(self):
        model_list = {
            "models": {
                "existing": {"1b": {"name": "keep"}},
            }
        }
        model_info = {"existing:1b": [{"path": "keep"}]}
        entry = {"name": "generated"}
        records = [{"path": "model.onnx", "type": "file"}]

        updated_list, updated_info = package_tool.updated_catalog_documents(
            model_list,
            model_info,
            entry,
            records,
        )

        self.assertEqual(
            updated_list["models"]["existing"],
            {"1b": {"name": "keep"}},
        )
        self.assertEqual(
            updated_info["existing:1b"],
            [{"path": "keep"}],
        )
        self.assertEqual(
            updated_list["models"]["phi4-mini-it-aie4"]["4b"],
            entry,
        )
        self.assertEqual(
            updated_info["phi4-mini-it-aie4:4b"],
            records,
        )

    def test_http_unauthorized_falls_back_to_pinned_git_metadata(self):
        unauthorized = HTTPError(
            package_tool.UPSTREAM_API_URL,
            401,
            "Unauthorized",
            {},
            None,
        )
        expected = [{"path": "model.onnx", "type": "file"}]
        with (
            patch.object(
                package_tool.urllib.request,
                "urlopen",
                side_effect=unauthorized,
            ),
            patch.object(
                package_tool,
                "git_metadata_records",
                return_value=expected,
            ) as git_records,
        ):
            actual = package_tool.huggingface_metadata_records(
                Path("metadata checkout"),
                package_tool.UPSTREAM_COMMIT,
            )

        self.assertEqual(actual, expected)
        git_records.assert_called_once_with(
            Path("metadata checkout"),
            package_tool.UPSTREAM_COMMIT,
        )

    def test_http_metadata_is_normalized_to_git_fallback_schema(self):
        response = io.BytesIO(
            json.dumps(
                [
                    {
                        "type": "file",
                        "oid": "1" * 40,
                        "size": 4,
                        "lfs": {
                            "oid": "a" * 64,
                            "size": 4,
                            "pointerSize": 127,
                        },
                        "xetHash": "environment-specific",
                        "path": "model.onnx",
                    }
                ]
            ).encode()
        )
        with patch.object(
            package_tool.urllib.request,
            "urlopen",
            return_value=response,
        ):
            records = package_tool.huggingface_metadata_records(
                Path("unused"),
                package_tool.UPSTREAM_COMMIT,
            )

        self.assertEqual(
            records,
            [
                {
                    "type": "file",
                    "oid": "1" * 40,
                    "size": 4,
                    "lfs": {
                        "oid": "a" * 64,
                        "size": 4,
                        "pointerSize": 127,
                    },
                    "path": "model.onnx",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
