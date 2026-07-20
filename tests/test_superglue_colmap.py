from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
