from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "colmap" / "superglue_colmap.py"
SPEC = importlib.util.spec_from_file_location("superglue_colmap_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sg
SPEC.loader.exec_module(sg)


class ParserDefaultsTests(unittest.TestCase):
    def test_cycle_filter_is_opt_in(self) -> None:
        args = sg.build_arg_parser().parse_args([])
        self.assertFalse(args.cycle_filter)

    def test_registration_rescue_is_enabled(self) -> None:
        args = sg.build_arg_parser().parse_args([])
        self.assertTrue(args.registration_rescue)
        self.assertEqual(args.registration_rescue_hops, 2)


class CameraMajorInputTests(unittest.TestCase):
    @staticmethod
    def _touch(folder: Path, names: list[str]) -> None:
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_bytes(b"test")

    def test_transposes_matching_frame_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "Camera_2", ["1.jpg", "2.jpg"])
            self._touch(root / "Camera_10", ["1.png", "2.png"])

            frames, mapping = sg.discover_camera_major_frames(root)

            self.assertEqual([frame.name for frame in frames], ["1", "2"])
            self.assertEqual(mapping, {"1.png": "Camera_2", "2.png": "Camera_10"})
            self.assertEqual(
                [(item.path.stem, item.output_name) for item in frames[1].images],
                [("2", "1.png"), ("2", "2.png")],
            )

    def test_rejects_equal_counts_with_misaligned_frame_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "Camera_1", ["1.png", "3.png"])
            self._touch(root / "Camera_2", ["1.png", "2.png"])

            with self.assertRaisesRegex(ValueError, "same frame stems"):
                sg.discover_camera_major_frames(root)

    def test_flat_image_root_is_accepted_as_one_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "camera_1.png").write_bytes(b"test")
            (root / "camera_2.png").write_bytes(b"test")

            frames = sg.discover_frame_dirs(root)

            self.assertEqual(frames, [root])


class ManualCameraLayoutTests(unittest.TestCase):
    @staticmethod
    def _write_layout(path: Path, camera_ids: list[str], rows: list[list[dict | None]], mode=8, radius=1) -> None:
        path.write_text(
            json.dumps(
                {
                    "format": "camera-grid-layout",
                    "version": 2,
                    "cameras": [{"camera_id": camera_id} for camera_id in camera_ids],
                    "rows": rows,
                    "neighbour_settings": {"mode": mode, "radius": radius},
                }
            ),
            encoding="utf-8",
        )

    def test_json_layout_builds_same_position_and_external_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_grid.json"
            self._write_layout(
                path,
                ["wide.png", "tele.png", "next.png"],
                [[
                    {"enabled": True, "camera_ids": ["wide.png", "tele.png"]},
                    {"enabled": True, "camera_ids": ["next.png"]},
                ]],
                mode=4,
            )

            layout = sg.load_manual_camera_layout(path)
            pairs = set(sg.build_manual_camera_pairs(
                ["wide.png", "tele.png", "next.png"], layout
            ))

            self.assertEqual(layout.neighbour_mode, 4)
            self.assertEqual(len(layout.placed_ids), 3)
            self.assertEqual(
                sg.manual_colocated_images(["wide.png", "tele.png", "next.png"], layout),
                {"wide.png": ("tele.png",), "tele.png": ("wide.png",)},
            )
            self.assertEqual(
                pairs,
                {("wide.png", "tele.png"), ("wide.png", "next.png"), ("tele.png", "next.png")},
            )

    def test_version_one_single_camera_slots_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            data = {
                "format": "camera-grid-layout",
                "version": 1,
                "cameras": [{"camera_id": "a.png"}, {"camera_id": "b.png"}],
                "rows": [[
                    {"enabled": True, "camera_id": "a.png"},
                    {"enabled": True, "camera_id": "b.png"},
                ]],
                "neighbour_settings": {"mode": 4, "radius": 1},
            }
            path.write_text(json.dumps(data), encoding="utf-8")

            layout = sg.load_manual_camera_layout(path)

            self.assertEqual(layout.pairs, (("a.png", "b.png"),))

    def test_numeric_frame_prefix_is_ignored_when_uniquely_mapping_cameras(self) -> None:
        layout = sg.ManualCameraLayout(
            Path("manual.json"),
            ("000000-00-color.png", "000000-01-color.png"),
            ("000000-00-color.png", "000000-01-color.png"),
            (("000000-00-color.png", "000000-01-color.png"),),
        )

        pairs = sg.build_manual_camera_pairs(
            ["000123-00-color.png", "000123-01-color.png"], layout
        )

        self.assertEqual(pairs, [("000123-00-color.png", "000123-01-color.png")])

    def test_camera_major_generated_names_use_camera_folder_mapping(self) -> None:
        layout = sg.ManualCameraLayout(
            Path("manual.json"),
            ("Camera_2", "Camera_10"),
            ("Camera_2", "Camera_10"),
            (("Camera_2", "Camera_10"),),
        )

        pairs = sg.build_manual_camera_pairs(
            ["1.png", "2.png"],
            layout,
            {"1.png": "Camera_2", "2.png": "Camera_10"},
        )

        self.assertEqual(pairs, [("1.png", "2.png")])

    def test_uncovered_image_is_strict_by_default_with_explicit_fallback(self) -> None:
        layout = sg.ManualCameraLayout(
            Path("manual.txt"),
            ("a.png", "b.png"),
            ("a.png", "b.png"),
            (("a.png", "b.png"),),
        )

        with self.assertRaisesRegex(ValueError, "without any neighbour"):
            sg.build_manual_camera_pairs(["a.png", "b.png", "c.png"], layout)

        pairs = sg.build_manual_camera_pairs(
            ["a.png", "b.png", "c.png"], layout, unlisted_policy="exhaustive"
        )
        self.assertEqual(set(pairs), {("a.png", "b.png"), ("a.png", "c.png"), ("b.png", "c.png")})

    def test_exported_pair_text_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_grid_neighbour_pairs.txt"
            path.write_text("# generated\na.png\tb.png\nb.png\tc.png\n", encoding="utf-8")

            layout = sg.load_manual_camera_layout(path)

            self.assertEqual(layout.camera_ids, ("a.png", "b.png", "c.png"))
            self.assertEqual(layout.pairs, (("a.png", "b.png"), ("b.png", "c.png")))

    def test_registration_rescue_adds_only_wider_registered_neighbours(self) -> None:
        names = ["a.png", "b.png", "c.png", "d.png", "e.png"]
        base = [("a.png", "b.png"), ("b.png", "c.png"), ("c.png", "d.png"), ("d.png", "e.png")]

        pairs = sg.build_registration_rescue_pairs(
            names,
            base,
            ["a.png"],
            {"b.png", "c.png", "d.png", "e.png"},
            hops=3,
            max_pairs_per_image=2,
        )

        self.assertEqual(pairs, [("a.png", "c.png"), ("a.png", "d.png")])


class FeatureMaskTests(unittest.TestCase):
    def test_lower_resolution_mask_uses_scaled_coordinates(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        feat = {
            "keypoints": [torch.tensor([[0.6, 1.0], [3.0, 1.0]], dtype=torch.float32)],
            "scores": [torch.tensor([0.1, 0.9], dtype=torch.float32)],
            "descriptors": [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)],
            "scales": (2.0, 2.0),
            "shape": (4, 4),
        }
        mask = np.array([[0, 255], [0, 255]], dtype=np.uint8)

        filtered = sg.apply_feature_mask(feat, mask)

        np.testing.assert_allclose(filtered["keypoints"][0].cpu().numpy(), [[3.0, 1.0]])
        np.testing.assert_allclose(filtered["scores"][0].cpu().numpy(), [0.9])
        np.testing.assert_allclose(filtered["descriptors"][0].cpu().numpy(), [[2.0], [4.0]])


class MatchIndexingTests(unittest.TestCase):
    @staticmethod
    def _reference_index(image_names, pair_matches, quantization):
        key_maps = {name: {} for name in image_names}
        keypoints = {name: [] for name in image_names}
        indexed_pairs = []
        q = max(float(quantization), 1e-6)
        for pair in pair_matches:
            indices = []
            for x0, y0, x1, y1 in pair.matches:
                endpoints = ((pair.name0, x0, y0), (pair.name1, x1, y1))
                pair_indices = []
                for name, x, y in endpoints:
                    key = (int(round(float(x) / q)), int(round(float(y) / q)))
                    if key not in key_maps[name]:
                        key_maps[name][key] = len(keypoints[name])
                        keypoints[name].append((float(x), float(y)))
                    pair_indices.append(key_maps[name][key])
                indices.append(tuple(pair_indices))
            indexed_pairs.append(
                (pair.name0, pair.name1, np.unique(np.asarray(indices, dtype=np.uint32), axis=0))
            )
        return {
            name: np.asarray(points, dtype=np.float32).reshape(-1, 2)
            for name, points in keypoints.items()
        }, indexed_pairs

    def test_vectorized_indexing_preserves_reference_correspondences(self) -> None:
        pairs = [
            sg.PairMatch(
                "a.png",
                "b.png",
                np.array(
                    [[1.01, 2.01, 4.0, 5.0], [1.02, 2.02, 6.0, 7.0]],
                    dtype=np.float32,
                ),
            ),
            sg.PairMatch(
                "a.png",
                "c.png",
                np.array([[1.0, 2.0, 8.0, 9.0], [3.0, 3.0, 10.0, 11.0]], dtype=np.float32),
            ),
        ]
        image_names = ["a.png", "b.png", "c.png"]
        expected_points, expected_pairs = self._reference_index(image_names, pairs, 0.25)
        actual_points, actual_pairs = sg.index_pair_matches(
            {name: object() for name in image_names}, pairs, 0.25
        )

        def point_map(points):
            return {
                tuple(np.round(point.astype(np.float64) / 0.25).astype(np.int64)): tuple(point)
                for point in points
            }

        for name in image_names:
            self.assertEqual(point_map(actual_points[name]), point_map(expected_points[name]))

        def correspondence_keys(points, indexed):
            result = []
            for name0, name1, matches in indexed:
                keys = {
                    (
                        tuple(np.round(points[name0][i0].astype(np.float64) / 0.25).astype(np.int64)),
                        tuple(np.round(points[name1][i1].astype(np.float64) / 0.25).astype(np.int64)),
                    )
                    for i0, i1 in matches
                }
                result.append((name0, name1, keys))
            return result

        self.assertEqual(
            correspondence_keys(actual_points, actual_pairs),
            correspondence_keys(expected_points, expected_pairs),
        )


class RegistrationRescueModelTests(unittest.TestCase):
    def test_appended_database_keypoints_are_mirrored_into_text_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "cameras.txt").write_text("# cameras\n", encoding="utf-8")
            (source / "points3D.txt").write_text("# points\n", encoding="utf-8")
            (source / "images.txt").write_text(
                "# images\n1 1 0 0 0 0 0 0 1 a.png\n10 20 -1\n",
                encoding="utf-8",
            )
            database = root / "database.db"
            con = sqlite3.connect(database)
            try:
                con.execute(
                    "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB)"
                )
                points = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
                con.execute(
                    "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (1, 2, 2, ?)",
                    (points.tobytes(),),
                )
                con.commit()
            finally:
                con.close()

            output = sg.prepare_model_for_appended_keypoints(source, database, root / "output")
            lines = (output / "images.txt").read_text(encoding="utf-8").splitlines()

            self.assertEqual(lines[-1], "10 20 -1 30 40 -1")


if __name__ == "__main__":
    unittest.main()
