from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from tools.generate_phi4_corelib_manifest import (
    MAX_U64,
    _generate_manifest,
    expected_matmul_components,
    host_role,
    matmul_roles,
    required_initializer_roles,
    ssmlp_roles,
)


class ManifestGeneratorTests(unittest.TestCase):
    def _write_model(
        self,
        root: Path,
        initializers: list[TensorProto],
    ) -> Path:
        graph = helper.make_graph(
            nodes=[],
            name="synthetic-phi4",
            inputs=[],
            outputs=[],
            initializer=initializers,
        )
        model = helper.make_model(graph)
        path = root / "model.onnx"
        path.write_bytes(model.SerializeToString())
        return path

    def _embedded(
        self,
        name: str,
        data_type: int,
        shape: list[int],
        raw_data: bytes,
    ) -> TensorProto:
        tensor = TensorProto()
        tensor.name = name
        tensor.data_type = data_type
        tensor.dims.extend(shape)
        tensor.raw_data = raw_data
        return tensor

    def _external(
        self,
        name: str,
        data_type: int,
        shape: list[int],
        metadata: list[tuple[str, str]],
    ) -> TensorProto:
        tensor = TensorProto()
        tensor.name = name
        tensor.data_type = data_type
        tensor.dims.extend(shape)
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in metadata:
            item = tensor.external_data.add()
            item.key = key
            item.value = value
        return tensor

    def _exact_roles(
        self,
        name: str,
        *,
        dtypes: set[str] | None = None,
        shape: list[int] | None = None,
    ) -> dict[str, dict[str, object]]:
        return {
            name: {
                "role": "test.tensor",
                "dtypes": dtypes or {"uint8"},
                "shape": shape or [2, 2],
            }
        }

    def test_matmul_component_shapes(self):
        expected = expected_matmul_components(k=3072, n=1024, group=128)
        self.assertEqual(expected["qweight"], ("uint8", [1024, 1536]))
        self.assertEqual(
            expected["scales"],
            ({"float16", "float32"}, [1024, 24]),
        )
        self.assertEqual(expected["qzeros"], ("uint8", [1024, 12]))

        odd_group_count = expected_matmul_components(
            k=384,
            n=5,
            group=128,
        )
        self.assertEqual(odd_group_count["qzeros"], ("uint8", [5, 2]))

    def test_matmul_roles_use_exact_names_and_constraints(self):
        prefix = "model.layers.7.attn.k_proj.MatMulNBits"
        self.assertEqual(
            matmul_roles(prefix, 3072, 1024, 128),
            {
                f"{prefix}.qweight": {
                    "role": "matmul.qweight",
                    "weight_object": prefix,
                    "dtypes": {"uint8"},
                    "shape": [1024, 1536],
                },
                f"{prefix}.scales": {
                    "role": "matmul.scales",
                    "weight_object": prefix,
                    "dtypes": {"float16", "float32"},
                    "shape": [1024, 24],
                },
                f"{prefix}.qzeros": {
                    "role": "matmul.qzeros",
                    "weight_object": prefix,
                    "dtypes": {"uint8"},
                    "shape": [1024, 12],
                },
            },
        )

    def test_ssmlp_gate_up_down_and_norm_shapes(self):
        roles = ssmlp_roles(0)
        base = "model.layers.0"
        group = f"{base}.ssmlp"

        self.assertEqual(
            roles[f"{base}.mlp.gate_proj.MatMulNBits.qweight"],
            {
                "role": "ssmlp.gate.qweight",
                "weight_object": group,
                "dtypes": {"uint8"},
                "shape": [8192, 1536],
            },
        )
        self.assertEqual(
            roles[f"{base}.mlp.up_proj.MatMulNBits.scales"]["shape"],
            [8192, 24],
        )
        self.assertEqual(
            roles[f"{base}.mlp.up_proj.MatMulNBits.qzeros"]["shape"],
            [8192, 12],
        )
        self.assertEqual(
            roles[f"{base}.mlp.down_proj.MatMulNBits.qweight"]["shape"],
            [3072, 4096],
        )
        self.assertEqual(
            roles[f"{base}.mlp.down_proj.MatMulNBits.scales"]["shape"],
            [3072, 64],
        )
        self.assertEqual(
            roles[f"{base}.mlp.down_proj.MatMulNBits.qzeros"]["shape"],
            [3072, 32],
        )
        self.assertEqual(
            roles[f"{base}.post_attention_layernorm.weight"],
            {
                "role": "ssmlp.norm0",
                "weight_object": group,
                "dtypes": {"float16", "float32"},
                "shape": [3072],
            },
        )
        self.assertEqual(
            roles["model.layers.1.input_layernorm.weight"]["role"],
            "ssmlp.norm1",
        )

    def test_required_roles_cover_exact_32_layer_driver_map(self):
        roles = required_initializer_roles()
        self.assertEqual(len(roles), 743)

        groups = {
            record["weight_object"]
            for record in roles.values()
            if "weight_object" in record
        }
        self.assertEqual(len(groups), 161)
        expected_attention_groups = {
            f"model.layers.{layer}.attn.{projection}.MatMulNBits"
            for layer in range(32)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        self.assertTrue(expected_attention_groups.issubset(groups))
        self.assertEqual(
            {
                group
                for group in groups
                if str(group).endswith(".ssmlp")
            },
            {f"model.layers.{layer}.ssmlp" for layer in range(32)},
        )
        self.assertIn("lm_head.MatMulNBits", groups)

        o_prefix = "model.layers.31.attn.o_proj.MatMulNBits"
        self.assertEqual(
            roles[f"{o_prefix}.qweight"]["shape"],
            [3072, 1536],
        )
        self.assertNotIn(
            "model.layers.32.attn.q_proj.MatMulNBits.qweight",
            roles,
        )

    def test_layer_31_uses_phantom_layer_32_final_norm(self):
        roles = ssmlp_roles(31)
        final_norm = "model.layers.32.final_norm_layernorm.weight"
        self.assertIn(final_norm, roles)
        self.assertEqual(roles[final_norm]["role"], "ssmlp.norm1")
        self.assertNotIn("model.layers.32.input_layernorm.weight", roles)
        self.assertNotIn("model.norm.weight", required_initializer_roles())

    def test_host_tensor_constraints(self):
        self.assertEqual(
            host_role("embedding"),
            {
                "role": "embedding",
                "dtypes": {"float16"},
                "shape": [200064, 3072],
            },
        )
        self.assertEqual(
            host_role("input_norm"),
            {
                "role": "input_norm",
                "dtypes": {"float16", "float32"},
                "shape": [3072],
            },
        )
        for role in ("cos_cache", "sin_cache"):
            self.assertEqual(
                host_role(role),
                {
                    "role": role,
                    "dtypes": {"float16", "float32"},
                    "rank": 2,
                    "minimum_shape": [4096, 48],
                },
            )

    def test_external_initializer_metadata_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "weights" / "data.bin"
            data_path.parent.mkdir()
            data_path.write_bytes(b"HEADER" + bytes([1, 2, 3, 4]) + b"TAIL")
            tensor = self._external(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                [
                    ("location", "weights/data.bin"),
                    ("offset", "6"),
                    ("length", "4"),
                ],
            )
            model_path = self._write_model(root, [tensor])
            output = root / "manifest.json"

            manifest = _generate_manifest(
                root,
                output,
                True,
                self._exact_roles("test.weight"),
            )

            self.assertEqual(
                manifest["initializers"]["test.weight"],
                {
                    "dtype": "uint8",
                    "file": "weights/data.bin",
                    "length": 4,
                    "offset": 6,
                    "role": "test.tensor",
                    "shape": [2, 2],
                },
            )
            self.assertEqual(
                set(manifest["files"]),
                {"model.onnx", "weights/data.bin"},
            )
            self.assertEqual(
                manifest["files"]["weights/data.bin"],
                {
                    "size": data_path.stat().st_size,
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                },
            )
            self.assertEqual(
                manifest["files"]["model.onnx"]["sha256"],
                hashlib.sha256(model_path.read_bytes()).hexdigest(),
            )

    def test_full_hash_flag_controls_sha256_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = self._embedded(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                b"data",
            )
            self._write_model(root, [tensor])

            manifest = _generate_manifest(
                root,
                root / "manifest.json",
                False,
                self._exact_roles("test.weight"),
            )

            self.assertTrue(manifest["files"])
            for record in manifest["files"].values():
                self.assertNotIn("sha256", record)

    def test_initializer_schema_omits_validation_only_group_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = self._embedded(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                b"data",
            )
            self._write_model(root, [tensor])
            roles = self._exact_roles("test.weight")
            roles["test.weight"]["weight_object"] = "test.matmul"

            manifest = _generate_manifest(
                root,
                root / "manifest.json",
                False,
                roles,
            )

            self.assertEqual(
                manifest["initializers"]["test.weight"],
                {
                    "dtype": "uint8",
                    "file": "corelib_embedded_initializers.bin",
                    "length": 4,
                    "offset": 0,
                    "role": "test.tensor",
                    "shape": [2, 2],
                },
            )

    def test_embedded_initializers_are_sorted_into_one_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            z_tensor = self._embedded(
                "z.tensor",
                TensorProto.UINT8,
                [3],
                b"XYZ",
            )
            a_tensor = self._embedded(
                "a.tensor",
                TensorProto.UINT8,
                [2],
                b"ab",
            )
            self._write_model(root, [z_tensor, a_tensor])
            roles = {
                "z.tensor": {
                    "role": "test.z",
                    "dtypes": {"uint8"},
                    "shape": [3],
                },
                "a.tensor": {
                    "role": "test.a",
                    "dtypes": {"uint8"},
                    "shape": [2],
                },
            }

            first = root / "first.json"
            second = root / "second.json"
            manifest = _generate_manifest(root, first, True, roles)
            _generate_manifest(root, second, True, roles)

            sidecar = root / "corelib_embedded_initializers.bin"
            self.assertEqual(sidecar.read_bytes(), b"abXYZ")
            self.assertEqual(
                manifest["initializers"]["a.tensor"]["offset"],
                0,
            )
            self.assertEqual(
                manifest["initializers"]["z.tensor"]["offset"],
                2,
            )
            self.assertEqual(
                manifest["files"]["corelib_embedded_initializers.bin"],
                {
                    "size": 5,
                    "sha256": hashlib.sha256(b"abXYZ").hexdigest(),
                },
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())
            decoded = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(
                list(decoded["initializers"]),
                ["a.tensor", "z.tensor"],
            )

    def test_wider_and_longer_rope_source_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = 4097
            columns = 64
            tensor = self._embedded(
                "cos_cache",
                TensorProto.FLOAT16,
                [rows, columns],
                bytes(rows * columns * 2),
            )
            self._write_model(root, [tensor])

            manifest = _generate_manifest(
                root,
                root / "manifest.json",
                False,
                {"cos_cache": host_role("cos_cache")},
            )

            self.assertEqual(
                manifest["initializers"]["cos_cache"]["shape"],
                [4097, 64],
            )
            self.assertEqual(
                manifest["initializers"]["cos_cache"]["dtype"],
                "float16",
            )

    def test_path_traversal_and_absolute_locations_are_rejected(self):
        bad_locations = (
            "../escape.bin",
            "safe/../../escape.bin",
            r"safe\..\escape.bin",
            "/absolute.bin",
            "C:/absolute.bin",
            r"\\server\share\absolute.bin",
        )
        for location in bad_locations:
            with self.subTest(location=location):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    tensor = self._external(
                        "test.weight",
                        TensorProto.UINT8,
                        [2, 2],
                        [
                            ("location", location),
                            ("offset", "0"),
                            ("length", "4"),
                        ],
                    )
                    self._write_model(root, [tensor])

                    with self.assertRaisesRegex(ValueError, "path|location"):
                        _generate_manifest(
                            root,
                            root / "manifest.json",
                            False,
                            self._exact_roles("test.weight"),
                        )

    def test_missing_length_and_range_overflow_are_rejected(self):
        cases = (
            (
                [
                    ("location", "weights.bin"),
                    ("offset", "0"),
                ],
                "length",
            ),
            (
                [
                    ("location", "weights.bin"),
                    ("offset", str(MAX_U64)),
                    ("length", "2"),
                ],
                "overflow",
            ),
        )
        for metadata, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "weights.bin").write_bytes(b"data")
                    tensor = self._external(
                        "test.weight",
                        TensorProto.UINT8,
                        [2, 2],
                        metadata,
                    )
                    self._write_model(root, [tensor])

                    with self.assertRaisesRegex(ValueError, message):
                        _generate_manifest(
                            root,
                            root / "manifest.json",
                            False,
                            self._exact_roles("test.weight"),
                        )

    def test_wrong_dtype_rank_and_shape_are_rejected(self):
        cases = (
            (
                TensorProto.FLOAT16,
                [2, 2],
                bytes(8),
                "dtype",
            ),
            (
                TensorProto.UINT8,
                [4],
                bytes(4),
                "shape",
            ),
            (
                TensorProto.UINT8,
                [2, 3],
                bytes(6),
                "shape",
            ),
        )
        for data_type, shape, raw_data, message in cases:
            with self.subTest(data_type=data_type, shape=shape):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    tensor = self._embedded(
                        "test.weight",
                        data_type,
                        shape,
                        raw_data,
                    )
                    self._write_model(root, [tensor])

                    with self.assertRaisesRegex(ValueError, message):
                        _generate_manifest(
                            root,
                            root / "manifest.json",
                            False,
                            self._exact_roles("test.weight"),
                        )

    def test_byte_count_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embedded = self._embedded(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                b"abc",
            )
            self._write_model(root, [embedded])
            with self.assertRaisesRegex(ValueError, "byte count"):
                _generate_manifest(
                    root,
                    root / "embedded.json",
                    False,
                    self._exact_roles("test.weight"),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"abc")
            external = self._external(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                [
                    ("location", "weights.bin"),
                    ("offset", "0"),
                    ("length", "3"),
                ],
            )
            self._write_model(root, [external])
            with self.assertRaisesRegex(ValueError, "byte count"):
                _generate_manifest(
                    root,
                    root / "external.json",
                    False,
                    self._exact_roles("test.weight"),
                )

    def test_missing_and_duplicate_required_initializers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_model(root, [])
            with self.assertRaisesRegex(ValueError, "missing initializer"):
                _generate_manifest(
                    root,
                    root / "missing.json",
                    False,
                    self._exact_roles("test.weight"),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._embedded(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                b"abcd",
            )
            second = self._embedded(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                b"efgh",
            )
            self._write_model(root, [first, second])
            with self.assertRaisesRegex(ValueError, "duplicate initializer"):
                _generate_manifest(
                    root,
                    root / "duplicate.json",
                    False,
                    self._exact_roles("test.weight"),
                )

    def test_manifest_has_locked_schema_and_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = self._embedded(
                "test.weight",
                TensorProto.UINT8,
                [2, 2],
                b"abcd",
            )
            self._write_model(root, [tensor])

            manifest = _generate_manifest(
                root,
                root / "manifest.json",
                False,
                self._exact_roles("test.weight"),
            )

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["execution_backend"], "corelib_aie4")
            self.assertEqual(
                manifest["model"],
                {
                    "family": "phi4",
                    "group_size": 128,
                    "head_size": 128,
                    "hidden_size": 3072,
                    "intermediate_size": 8192,
                    "kv_heads": 8,
                    "layers": 32,
                    "num_heads": 24,
                    "rms_epsilon": 0.00001,
                    "rope_dim": 96,
                    "vocab_size": 200064,
                },
            )
            self.assertEqual(manifest["backend"], {"max_seq": 4096})


if __name__ == "__main__":
    unittest.main()
