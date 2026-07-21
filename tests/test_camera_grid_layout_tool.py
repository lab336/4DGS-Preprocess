from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "colmap" / "camera_grid_layout_tool.py"
SPEC = importlib.util.spec_from_file_location("camera_grid_layout_tool_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def item(camera_id: str) -> tool.CameraItem:
    return tool.CameraItem(camera_id, f"C:/{camera_id}.png", f"C:/{camera_id}.png")


class CameraGridLayoutTests(unittest.TestCase):
    def test_place_move_and_group_without_duplicate_assignments(self) -> None:
        layout = tool.CameraGridLayout(2, 2)
        for camera_id in ("cam1", "cam2", "cam3"):
            layout.add_camera(item(camera_id))

        layout.place_camera("cam1", 0, 0)
        layout.place_camera("cam2", 0, 1)
        layout.place_camera("cam1", 0, 1)
        layout.place_camera("cam3", 0, 1)

        self.assertEqual(layout.rows[0][0].camera_ids, [])
        self.assertEqual(layout.rows[0][1].camera_ids, ["cam2", "cam1", "cam3"])
        layout.place_camera("cam1", 0, 1)
        self.assertEqual(layout.rows[0][1].camera_ids, ["cam2", "cam1", "cam3"])
        self.assertEqual(layout.placed_camera_ids(), {"cam1", "cam2", "cam3"})

    def test_per_row_length_and_physical_gap_return_camera_to_unplaced(self) -> None:
        layout = tool.CameraGridLayout(2, 3)
        layout.add_camera(item("cam1"))
        layout.add_camera(item("cam2"))
        layout.place_camera("cam1", 0, 2)
        layout.place_camera("cam2", 1, 1)

        layout.set_row_length(0, 2)
        removed = layout.toggle_slot_enabled(1, 1)

        self.assertEqual(removed, ("cam2",))
        self.assertEqual([len(row) for row in layout.rows], [2, 3])
        self.assertFalse(layout.rows[1][1].enabled)
        self.assertEqual(layout.placed_camera_ids(), set())
        self.assertEqual(set(layout.cameras), {"cam1", "cam2"})

    def test_serpentine_auto_fill_uses_natural_camera_order(self) -> None:
        layout = tool.CameraGridLayout(2, 3)
        for camera_id in ("cam10", "cam2", "cam1", "cam6", "cam5", "cam4"):
            layout.add_camera(item(camera_id))

        remaining = layout.auto_fill("serpentine")

        self.assertEqual(remaining, [])
        self.assertEqual([slot.camera_id for slot in layout.rows[0]], ["cam1", "cam2", "cam4"])
        self.assertEqual([slot.camera_id for slot in layout.rows[1]], ["cam10", "cam6", "cam5"])

    def test_four_and_eight_neighbour_graphs(self) -> None:
        layout = tool.CameraGridLayout(2, 3)
        for index, position in enumerate(layout.active_positions(), 1):
            camera_id = f"cam{index}"
            layout.add_camera(item(camera_id))
            layout.place_camera(camera_id, *position)

        four_pairs = layout.neighbour_pairs(4, 1)
        eight_pairs = layout.neighbour_pairs(8, 1)
        report = layout.validate(8, 1)

        self.assertEqual(len(four_pairs), 7)
        self.assertEqual(len(eight_pairs), 11)
        self.assertTrue(report.connected)
        self.assertEqual(len(report.components), 1)
        self.assertEqual(report.isolated, ())

    def test_four_neighbour_radius_does_not_add_diagonals(self) -> None:
        layout = tool.CameraGridLayout(3, 3)
        for camera_id, position in (
            ("center", (1, 1)),
            ("same_row", (1, 2)),
            ("same_column", (2, 1)),
            ("diagonal", (2, 2)),
        ):
            layout.add_camera(item(camera_id))
            layout.place_camera(camera_id, *position)

        four_pairs = set(layout.neighbour_pairs(4, 2))
        eight_pairs = set(layout.neighbour_pairs(8, 2))

        self.assertIn(("center", "same_row"), four_pairs)
        self.assertIn(("center", "same_column"), four_pairs)
        self.assertNotIn(("center", "diagonal"), four_pairs)
        self.assertIn(("center", "diagonal"), eight_pairs)

    def test_validation_finds_disconnected_and_isolated_cameras(self) -> None:
        layout = tool.CameraGridLayout(1, 5)
        for camera_id, column in (("left", 0), ("middle", 1), ("right", 4)):
            layout.add_camera(item(camera_id))
            layout.place_camera(camera_id, 0, column)

        report = layout.validate(4, 1)

        self.assertFalse(report.connected)
        self.assertEqual(len(report.components), 2)
        self.assertEqual(report.isolated, ("right",))

    def test_same_position_cameras_share_internal_and_external_neighbours(self) -> None:
        layout = tool.CameraGridLayout(1, 2)
        for camera_id in ("wide", "tele", "neighbour"):
            layout.add_camera(item(camera_id))
        layout.place_camera("wide", 0, 0)
        layout.place_camera("tele", 0, 0)
        layout.place_camera("neighbour", 0, 1)

        pairs = set(layout.neighbour_pairs(4, 1))
        report = layout.validate(4, 1)

        self.assertEqual(
            pairs,
            {("neighbour", "tele"), ("neighbour", "wide"), ("tele", "wide")},
        )
        self.assertEqual(report.placed, 3)
        self.assertEqual(report.occupied_positions, 2)
        self.assertEqual(report.multi_camera_positions, 1)
        self.assertTrue(report.connected)

    def test_round_trip_and_export_bundle(self) -> None:
        layout = tool.CameraGridLayout(2, 2)
        for camera_id, position in (
            ("cam1", (0, 0)),
            ("cam1-tele", (0, 0)),
            ("cam2", (0, 1)),
            ("cam3", (1, 0)),
        ):
            layout.add_camera(item(camera_id))
            layout.place_camera(camera_id, *position)
        layout.set_slot_enabled(1, 1, False)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "studio_grid.json"
            paths = tool.export_layout(layout, output, 8, 1)
            loaded = tool.CameraGridLayout.load(output)

            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(loaded.to_dict(), layout.to_dict())
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exported["version"], 2)
            self.assertEqual(len(exported["neighbour_pairs"]), 6)
            self.assertEqual(exported["validation"]["occupied_positions"], 3)
            self.assertEqual(exported["validation"]["multi_camera_positions"], 1)
            self.assertIn("cam1\tcam2", paths[3].read_text(encoding="utf-8"))
            self.assertIn("cam1 | cam1-tele", paths[1].read_text(encoding="utf-8"))
            self.assertIn("slot_camera_count", paths[2].read_text(encoding="utf-8-sig"))
            self.assertIn("-", paths[1].read_text(encoding="utf-8"))

    def test_version_one_layout_migrates_to_multi_camera_slots(self) -> None:
        layout = tool.CameraGridLayout(1, 2)
        layout.add_camera(item("legacy"))
        legacy = layout.to_dict()
        legacy["version"] = 1
        legacy["rows"] = [[
            {"enabled": True, "camera_id": "legacy"},
            {"enabled": True, "camera_id": None},
        ]]

        migrated = tool.CameraGridLayout.from_dict(legacy)

        self.assertEqual(migrated.rows[0][0].camera_ids, ["legacy"])
        self.assertEqual(migrated.to_dict()["version"], 2)
        self.assertIn("camera_ids", migrated.to_dict()["rows"][0][0])

    def test_malformed_layout_reports_layout_error(self) -> None:
        valid = tool.CameraGridLayout(1, 1).to_dict()
        malformed_cases = [
            [],
            {**valid, "version": "not-an-integer"},
            {**valid, "default_columns": 0},
            {**valid, "cameras": {}},
            {**valid, "rows": [[{"enabled": "false", "camera_ids": []}]]},
            {**valid, "rows": [[{"enabled": True, "camera_ids": ["same", "same"]}]]},
        ]

        for malformed in malformed_cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(tool.LayoutError):
                    tool.CameraGridLayout.from_dict(malformed)


class CameraDiscoveryTests(unittest.TestCase):
    def test_camera_folders_use_first_naturally_sorted_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder_name in ("Camera_10", "Camera_2"):
                folder = root / folder_name
                folder.mkdir()
                (folder / "10.png").write_bytes(b"preview")
                (folder / "2.png").write_bytes(b"preview")

            items = tool.discover_camera_items(root)

            self.assertEqual([entry.camera_id for entry in items], ["Camera_2", "Camera_10"])
            self.assertEqual([Path(entry.preview_path).name for entry in items], ["2.png", "2.png"])
            self.assertTrue(all(entry.source_kind == "camera_folder" for entry in items))


if __name__ == "__main__":
    unittest.main()
