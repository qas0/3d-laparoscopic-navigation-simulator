from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyvista as pv
from PIL import Image
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

from capture import CaptureFrame, SaveCapture, SnapshotScene


class CaptureTests(unittest.TestCase):
    def test_DepthAndClassesMatchTransformedSurfacesWithoutChangingLiveState(self):
        plane = pv.Plane(i_size=12, j_size=9, i_resolution=1, j_resolution=1)
        plane.point_data["rgb"] = np.tile([230, 40, 20], (plane.n_points, 1)).astype(np.uint8)
        plane.set_active_scalars("rgb")
        plane_mapper = vtkPolyDataMapper()
        plane_mapper.SetInputData(plane)
        plane_mapper.SetColorModeToDirectScalars()
        plane_mapper.ScalarVisibilityOn()
        plane_actor = vtkActor()
        plane_actor.SetMapper(plane_mapper)
        plane_actor.SetPosition(0.3, -0.4, 0.7)
        plane_actor.GetProperty().SetColor(0.2, 0.6, 0.3)
        plane_actor.GetProperty().SetSpecular(0.35)
        plane_actor.GetProperty().SetSpecularPower(24)

        box = pv.Cube(bounds=(-1.5, 1.5, -1.5, 1.5, 3, 5)).triangulate()
        box_mapper = vtkPolyDataMapper()
        box_mapper.SetInputData(box)
        box_actor = vtkActor()
        box_actor.SetMapper(box_mapper)
        box_actor.SetPosition(0.3, -0.4, 0)
        box_actor.GetProperty().SetColor(0.3, 0.4, 0.8)
        actors = {"tissue": [plane_actor], "instrument": [box_actor]}
        camera = pv.Camera()
        camera.position = (0, 0, 12)
        camera.focal_point = (0, 0, 0)
        camera.up = (0, 1, 0)
        camera.view_angle = 60
        camera.clipping_range = (0.5, 40)
        camera.SetWindowCenter(0.12, -0.08)
        light = pv.Light(position=(0, 0, 12), focal_point=(0, 0, 0), light_type="scene light")
        metadata = {"target_world_mm": np.array([0.3, -0.4, 0.7]), "scenario": "synthetic"}
        points_before = plane.points.copy()
        scalars_before = plane.point_data["rgb"].copy()
        camera_before = (
            camera.position, camera.focal_point, camera.up, camera.view_angle,
            camera.clipping_range, camera.GetWindowCenter(), camera.GetMTime(),
        )
        actor_before = (
            plane_actor.GetMapper(), plane_actor.GetProperty(), plane_actor.GetPosition(),
            plane_actor.GetVisibility(), plane_actor.GetProperty().GetColor(),
            plane_actor.GetProperty().GetSpecular(), plane_actor.GetProperty().GetSpecularPower(),
            plane_mapper.GetScalarVisibility(), plane_mapper.GetColorMode(),
        )
        light_before = (light.position, light.focal_point, light.intensity, light.GetMTime())

        rgb, labels = CaptureFrame(
            actors, camera, light, metadata, image_size=(160, 120),
            colour_overrides={"tissue": np.array([130, 215, 120]) / 255},
        )

        self.assertEqual(rgb.shape, (120, 160, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertGreater(np.ptp(rgb), 30)
        np.testing.assert_array_equal(labels["image_size"], [160, 120])
        np.testing.assert_array_equal(labels["class_ids"], [0, 1, 2])
        np.testing.assert_array_equal(labels["class_names"], ["background", "tissue", "instrument"])
        np.testing.assert_array_equal(labels["camera_clipping_mm"], [0.5, 40])
        mask = labels["semantic_mask"]
        depth = labels["depth_mm"]
        self.assertEqual(mask.shape, (120, 160))
        self.assertEqual(mask.dtype, np.uint8)
        np.testing.assert_array_equal(np.unique(mask), [0, 1, 2])
        np.testing.assert_array_equal(np.isfinite(depth), mask != 0)
        self.assertTrue(np.all(depth[mask != 0] > 0))

        # reconstruct visible points to check calibration and depth together
        rows, columns = np.nonzero(mask)
        rows, columns = rows[::17], columns[::17]
        pixel_rays = np.linalg.inv(labels["camera_intrinsics"]) @ np.vstack(
            [columns, rows, np.ones(len(rows))]
        )
        camera_points = pixel_rays * depth[rows, columns]
        world_points = np.linalg.inv(labels["world_to_camera"]) @ np.vstack(
            [camera_points, np.ones(len(rows))]
        )
        tissue_points = world_points[:3, mask[rows, columns] == 1]
        tool_points = world_points[:3, mask[rows, columns] == 2]
        self.assertGreater(tissue_points.shape[1], 50)
        self.assertGreater(tool_points.shape[1], 20)
        np.testing.assert_allclose(tissue_points[2], 0.7, atol=0.002)
        np.testing.assert_allclose(tool_points[2], 5, atol=0.002)
        self.assertLessEqual(np.max(np.abs(tissue_points[0] - 0.3)), 6.01)
        self.assertLessEqual(np.max(np.abs(tissue_points[1] + 0.4)), 4.51)
        self.assertLessEqual(np.max(np.abs(tool_points[0] - 0.3)), 1.51)
        self.assertLessEqual(np.max(np.abs(tool_points[1] + 0.4)), 1.51)
        self.assertTrue(bool(labels["target_in_frame"]))
        self.assertFalse(bool(labels["target_visible"]))

        np.testing.assert_array_equal(plane.points, points_before)
        np.testing.assert_array_equal(plane.point_data["rgb"], scalars_before)
        self.assertEqual(actor_before, (
            plane_actor.GetMapper(), plane_actor.GetProperty(), plane_actor.GetPosition(),
            plane_actor.GetVisibility(), plane_actor.GetProperty().GetColor(),
            plane_actor.GetProperty().GetSpecular(), plane_actor.GetProperty().GetSpecularPower(),
            plane_mapper.GetScalarVisibility(), plane_mapper.GetColorMode(),
        ))
        self.assertEqual(camera_before, (
            camera.position, camera.focal_point, camera.up, camera.view_angle,
            camera.clipping_range, camera.GetWindowCenter(), camera.GetMTime(),
        ))
        self.assertEqual(light_before, (light.position, light.focal_point, light.intensity, light.GetMTime()))
        self.assertEqual(set(metadata), {"target_world_mm", "scenario"})
        self.assertTrue(all(not value.dtype.hasobject for value in labels.values()))

    def test_TargetVisibilityUsesCameraProjectionAndIgnoresHiddenActors(self):
        plane = pv.Plane(i_size=12, j_size=9, i_resolution=1, j_resolution=1)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(plane)
        actor = vtkActor()
        actor.SetMapper(mapper)
        hidden_mapper = vtkPolyDataMapper()
        hidden_mapper.SetInputData(pv.Cube(bounds=(-3, 3, -3, 3, 3, 5)).triangulate())
        hidden = vtkActor()
        hidden.SetMapper(hidden_mapper)
        hidden.VisibilityOff()
        camera = pv.Camera()
        camera.position = (0, 0, 12)
        camera.focal_point = (0, 0, 0)
        camera.up = (0, 1, 0)
        camera.view_angle = 60
        camera.clipping_range = (0.5, 40)
        light = pv.Light(position=(0, 0, 12), focal_point=(0, 0, 0), light_type="scene light")
        actors = {"tissue": [actor], "hidden": [hidden]}
        target = np.array([2, 1, 0])

        _, labels = CaptureFrame(actors, camera, light, {"target_world_mm": target}, image_size=(160, 120))

        focal_length = 120 / (2 * np.tan(np.radians(60) / 2))
        np.testing.assert_allclose(labels["target_pixel_uv"],
                                   [79.5 + focal_length * 2 / 12, 59.5 - focal_length / 12], atol=1e-6)
        self.assertAlmostEqual(float(labels["target_camera_z_mm"]), 12)
        self.assertTrue(bool(labels["target_in_frame"]))
        self.assertTrue(bool(labels["target_visible"]))
        self.assertNotIn(2, np.unique(labels["semantic_mask"]))
        self.assertFalse(hidden.GetVisibility())

        _, behind = CaptureFrame(actors, camera, light, {"target_world_mm": [0, 0, 15]}, image_size=(160, 120))
        self.assertLess(float(behind["target_camera_z_mm"]), 0)
        self.assertFalse(bool(behind["target_in_frame"]))
        self.assertFalse(bool(behind["target_visible"]))

    def test_NearClippingHidesLensGeometryButKeepsRealOccluders(self):
        tissue_mapper = vtkPolyDataMapper()
        tissue_mapper.SetInputData(pv.Plane(i_size=12, j_size=9, i_resolution=1, j_resolution=1))
        tissue = vtkActor()
        tissue.SetMapper(tissue_mapper)
        lens_mapper = vtkPolyDataMapper()
        lens_mapper.SetInputData(pv.Plane(center=(0, 0, 11.75), i_size=4, j_size=4,
                                        i_resolution=1, j_resolution=1))
        lens = vtkActor()
        lens.SetMapper(lens_mapper)
        camera = pv.Camera()
        camera.position = (0, 0, 12)
        camera.focal_point = (0, 0, 0)
        camera.up = (0, 1, 0)
        camera.view_angle = 60
        camera.clipping_range = (0.5, 40)
        light = pv.Light(position=(0, 0, 12), focal_point=(0, 0, 0), light_type="scene light")
        actors = {"tissue": [tissue], "lens": [lens]}
        metadata = {"target_world_mm": [2, 1, 0]}

        _, clipped = CaptureFrame(actors, camera, light, metadata, image_size=(160, 120))

        np.testing.assert_array_equal(clipped["camera_clipping_mm"], [0.5, 40])
        self.assertNotIn(2, np.unique(clipped["semantic_mask"]))
        self.assertIn(1, np.unique(clipped["semantic_mask"]))
        np.testing.assert_allclose(clipped["depth_mm"][clipped["semantic_mask"] == 1], 12, atol=0.002)
        self.assertTrue(bool(clipped["target_visible"]))
        self.assertTrue(lens.GetVisibility())

        # moves the same opaque surface beyond the near plane
        lens.SetPosition(0, 0, -0.5)
        _, occluded = CaptureFrame(actors, camera, light, metadata, image_size=(160, 120))

        np.testing.assert_array_equal(occluded["camera_clipping_mm"], [0.5, 40])
        self.assertIn(2, np.unique(occluded["semantic_mask"]))
        self.assertTrue(bool(occluded["target_in_frame"]))
        self.assertFalse(bool(occluded["target_visible"]))
        self.assertEqual(camera.clipping_range, (0.5, 40))

    def test_SnapshotFreezesGeometryTransformsAndMaterialTextures(self):
        mesh = pv.Plane(i_resolution=1, j_resolution=1)
        mesh.point_data["phase"] = np.arange(mesh.n_points, dtype=float)
        mesh.set_active_scalars("phase")
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        table = mapper.GetLookupTable()
        table.SetNumberOfTableValues(2)
        table.Build()
        table.SetTableValue(0, 0.2, 0.4, 0.6, 1)
        texture = pv.Texture(np.full((4, 4, 3), [140, 70, 35], dtype=np.uint8))
        texture.InterpolateOff()
        texture.SetUseSRGBColorSpace(True)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(2, 3, 4)
        actor.SetOrientation(15, 25, 35)
        actor.SetScale(1.2, 0.9, 0.7)
        actor.GetProperty().SetInterpolationToPBR()
        actor.GetProperty().SetMetallic(0.2)
        actor.GetProperty().SetRoughness(0.3)
        actor.GetProperty().SetBaseColorTexture(texture)
        camera = pv.Camera()
        camera.position = (1, 2, 12)
        camera.focal_point = (1, 2, 0)
        camera.clipping_range = (0.5, 40)
        light = pv.Light(position=(1, 2, 12), focal_point=(1, 2, 0), light_type="scene light")
        light.intensity = 0.7
        points_before = mesh.points.copy()
        scalars_before = mesh.point_data["phase"].copy()
        texture_before = pv.wrap(texture.GetInput()).point_data.active_scalars.copy()
        texture_references = texture.GetReferenceCount()
        matrix_before = np.array([[actor.GetMatrix().GetElement(row, column) for column in range(4)]
                                  for row in range(4)])
        table_before = table.GetTableValue(0)

        actors, frozen_camera, frozen_light = SnapshotScene({"tissue": [actor]}, camera, light)
        self.assertEqual(texture.GetReferenceCount(), texture_references)
        frozen = actors["tissue"][0]
        frozen_mesh = pv.wrap(frozen.GetMapper().GetInput())
        frozen_texture = frozen.GetProperty().GetTexture("albedoTex")
        mesh.points += [5, 6, 7]
        mesh.point_data["phase"][:] = 99
        actor.SetPosition(-10, -20, -30)
        actor.VisibilityOff()
        actor.GetProperty().SetMetallic(0.9)
        actor.GetProperty().SetRoughness(0.8)
        table.SetTableValue(0, 1, 0, 0, 1)
        texture.GetInput().GetPointData().GetScalars().SetTuple3(0, 1, 2, 3)
        texture.InterpolateOn()
        camera.position = (30, 40, 50)
        camera.clipping_range = (2, 200)
        light.position = (50, 60, 70)
        light.intensity = 1.5

        np.testing.assert_array_equal(frozen_mesh.points, points_before)
        np.testing.assert_array_equal(frozen_mesh.point_data["phase"], scalars_before)
        np.testing.assert_allclose(
            [[frozen.GetMatrix().GetElement(row, column) for column in range(4)] for row in range(4)],
            matrix_before,
        )
        self.assertTrue(frozen.GetVisibility())
        self.assertAlmostEqual(frozen.GetProperty().GetMetallic(), 0.2)
        self.assertAlmostEqual(frozen.GetProperty().GetRoughness(), 0.3)
        self.assertEqual(frozen.GetMapper().GetLookupTable().GetTableValue(0), table_before)
        np.testing.assert_array_equal(pv.wrap(frozen_texture.GetInput()).point_data.active_scalars,
                                      texture_before)
        self.assertFalse(frozen_texture.GetInterpolate())
        self.assertEqual(frozen_camera.position, (1, 2, 12))
        self.assertEqual(frozen_camera.clipping_range, (0.5, 40))
        self.assertEqual(frozen_light.position, (1, 2, 12))
        self.assertAlmostEqual(frozen_light.intensity, 0.7)

    def test_PickledSnapshotKeepsRenderedMaterialsDepthAndOffAxisCalibration(self):
        mesh = pv.Plane(i_size=10, j_size=8, i_resolution=2, j_resolution=2)
        mesh.lines = np.array([2, 0, 1])
        mesh.verts = np.array([1, 0])
        mesh.strips = np.array([3, 0, 1, 4])
        mesh.cell_data.clear()
        mesh.field_data["source"] = np.array(["tissue", "snapshot"])
        for association, count in ((mesh.point_data, mesh.n_points), (mesh.cell_data, mesh.n_cells)):
            association["phase"] = np.arange(count, dtype=np.float64)
            association["directions"] = np.tile([1., 0., 0.], (count, 1))
            association["tensor"] = np.tile(np.eye(3).ravel(), (count, 1))
            association["ids"] = np.arange(count, dtype=np.int64)
            association["enabled"] = np.arange(count) % 2 == 0
            association.SetActiveScalars("phase")
            association.SetActiveVectors("directions")
            association.SetActiveTensors("tensor")
            association.SetActiveGlobalIds("ids")
        mesh.point_data["tangents"] = np.tile([1., 0., 0.], (mesh.n_points, 1))
        mesh.GetPointData().SetActiveTangents("tangents")
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(0.5, -0.4, 0.7)
        actor.SetOrientation(0, 15, 10)
        actor.SetScale(1.1, 0.9, 1)
        texture_pixels = np.full((4, 4, 3), [140, 70, 35], dtype=np.uint8)
        texture_pixels[::2, ::2] = [70, 130, 90]
        texture = pv.Texture(texture_pixels)
        texture.SetUseSRGBColorSpace(True)
        actor.GetProperty().SetInterpolationToPBR()
        actor.GetProperty().SetMetallic(0.2)
        actor.GetProperty().SetRoughness(0.35)
        actor.GetProperty().SetBaseColorTexture(texture)
        camera = pv.Camera()
        camera.position = (0, 0, 12)
        camera.focal_point = (0, 0, 0)
        camera.up = (0, 1, 0)
        camera.view_angle = 57
        camera.clipping_range = (0.5, 40)
        camera.SetWindowCenter(-0.2, 0.13)
        light = pv.Light(position=(0, 0, 12), focal_point=(0, 0, 0), light_type="scene light")
        light.intensity = 0.7
        light.positional = True
        light.cone_angle = 65
        actors = {"tissue": [actor]}
        metadata = {"scenario": "pickled scene"}

        expected_rgb, expected_labels = CaptureFrame(actors, camera, light, metadata, image_size=(160, 120))
        payload = SnapshotScene(actors, camera, light, serializable=True)
        snapshot = pickle.loads(pickle.dumps(payload))
        rebuilt, _, _ = SnapshotScene(*snapshot)
        rebuilt_mesh = pv.wrap(rebuilt["tissue"][0].GetMapper().GetInput())
        for key in ("points", "faces", "lines", "verts", "strips"):
            self.assertIs(type(snapshot[0]["tissue"][0]["mesh"][key]), np.ndarray)
            np.testing.assert_array_equal(getattr(rebuilt_mesh, key), getattr(mesh, key))
        for name in ("point_data", "cell_data", "field_data"):
            before, after = getattr(mesh, name), getattr(rebuilt_mesh, name)
            self.assertEqual(list(after), list(before))
            for key, value in before.items():
                self.assertEqual(after[key].dtype, value.dtype)
                np.testing.assert_array_equal(after[key], value)
        for before, after in ((mesh.GetPointData(), rebuilt_mesh.GetPointData()),
                              (mesh.GetCellData(), rebuilt_mesh.GetCellData())):
            for role in range(before.NUM_ATTRIBUTES):
                expected = before.GetAbstractAttribute(role)
                actual = after.GetAbstractAttribute(role)
                self.assertEqual(None if actual is None else actual.GetName(),
                                 None if expected is None else expected.GetName())
        mesh.points += [5, 6, 7]
        mesh.point_data["phase"][:] = 99
        mesh.cell_data["phase"][:] = 99
        mesh.field_data["source"] = np.array(["changed"])
        actor.SetPosition(20, 30, 40)
        actor.GetProperty().SetRoughness(0.9)
        texture.GetInput().GetPointData().GetScalars().SetTuple3(0, 1, 2, 3)
        camera.SetWindowCenter(0, 0)
        camera.clipping_range = (2, 200)
        light.intensity = 0.1

        rgb, labels = CaptureFrame(*snapshot, metadata, image_size=(160, 120))

        np.testing.assert_array_equal(rgb, expected_rgb)
        self.assertEqual(set(labels), set(expected_labels))
        for name, expected in expected_labels.items():
            with self.subTest(label=name):
                np.testing.assert_equal(labels[name], expected)
        focal_length = 120 / (2 * np.tan(np.radians(57) / 2))
        np.testing.assert_allclose(labels["camera_intrinsics"],
                                   [[focal_length, 0, 95.5], [0, focal_length, 67.3], [0, 0, 1]])
        np.testing.assert_array_equal(labels["camera_clipping_mm"], [0.5, 40])

    def test_ColdCaptureProcessIncludesUntexturedInstrumentOnFirstFrame(self):
        instrument_mapper = vtkPolyDataMapper()
        instrument_mapper.SetInputData(pv.Cube(bounds=(-1.5, 1.5, -1.5, 1.5, 3, 5)).triangulate())
        instrument = vtkActor()
        instrument.SetMapper(instrument_mapper)
        instrument.GetProperty().SetColor(0.3, 0.4, 0.8)
        tissue_mapper = vtkPolyDataMapper()
        tissue_mapper.SetInputData(pv.Plane(i_size=12, j_size=9, i_resolution=1, j_resolution=1))
        tissue = vtkActor()
        tissue.SetMapper(tissue_mapper)
        texture = pv.Texture(np.full((4, 4, 3), [140, 70, 35], dtype=np.uint8))
        texture.SetUseSRGBColorSpace(True)
        tissue.GetProperty().SetInterpolationToPBR()
        tissue.GetProperty().SetRoughness(0.35)
        tissue.GetProperty().SetBaseColorTexture(texture)
        camera = pv.Camera()
        camera.position = (0, 0, 12)
        camera.focal_point = (0, 0, 0)
        camera.up = (0, 1, 0)
        camera.view_angle = 60
        camera.clipping_range = (0.5, 40)
        light = pv.Light(position=(0, 0, 12), focal_point=(0, 0, 0), light_type="scene light")
        snapshot = SnapshotScene({"instrument": [instrument], "tissue": [tissue]},
                                 camera, light, serializable=True)

        # starts a fresh interpreter so previous renders cannot initialize its factories
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            first_rgb, first_labels = executor.submit(
                CaptureFrame, *snapshot, {}, image_size=(160, 120)
            ).result(timeout=60)
            second_rgb, second_labels = executor.submit(
                CaptureFrame, *snapshot, {}, image_size=(160, 120)
            ).result(timeout=60)

        np.testing.assert_array_equal(np.unique(first_labels["semantic_mask"]), [0, 1, 2])
        np.testing.assert_array_equal(second_labels["semantic_mask"], first_labels["semantic_mask"])
        np.testing.assert_equal(second_labels["depth_mm"], first_labels["depth_mm"])
        np.testing.assert_allclose(second_rgb, first_rgb, rtol=0, atol=1)

    def test_SaveKeepsPairsAndSkipsExistingOrphanIndices(self):
        rgb = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        labels = {
            "schema_version": np.array(1),
            "class_names": np.array(["background", "tissue"]),
            "semantic_mask": np.ones((12, 16), dtype=np.uint8),
            "depth_mm": np.full((12, 16), 12, dtype=np.float32),
            "camera_intrinsics": np.eye(3),
        }
        with tempfile.TemporaryDirectory() as folder:
            image_folder = Path(folder) / "images"
            label_folder = Path(folder) / "labels"
            image_folder.mkdir()
            label_folder.mkdir()
            existing_image = image_folder / "frame_000001.png"
            existing_label = label_folder / "frame_000003.npz"
            existing_image.write_bytes(b"keep existing image")
            existing_label.write_bytes(b"keep existing labels")

            image_path, label_path = SaveCapture(folder, rgb, labels)
            second_image, second_label = SaveCapture(folder, rgb, labels)

            self.assertEqual(Path(image_path), image_folder / "frame_000004.png")
            self.assertEqual(Path(label_path), label_folder / "frame_000004.npz")
            self.assertEqual(Path(second_image), image_folder / "frame_000005.png")
            self.assertEqual(Path(second_label), label_folder / "frame_000005.npz")
            self.assertEqual(existing_image.read_bytes(), b"keep existing image")
            self.assertEqual(existing_label.read_bytes(), b"keep existing labels")
            with Image.open(image_path) as saved_image:
                np.testing.assert_array_equal(np.asarray(saved_image), rgb)
            with np.load(label_path, allow_pickle=False) as saved_labels:
                self.assertEqual(set(saved_labels.files), set(labels))
                for name, expected in labels.items():
                    np.testing.assert_array_equal(saved_labels[name], expected)
                    self.assertFalse(saved_labels[name].dtype.hasobject)

    def test_SaveFailureRemovesOnlyNewPartialFiles(self):
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            image_folder = Path(folder) / "images"
            image_folder.mkdir()
            existing = image_folder / "frame_000001.png"
            existing.write_bytes(b"keep existing image")
            with patch("capture.np.savez_compressed", side_effect=OSError("disk write failed")):
                with self.assertRaises(OSError):
                    SaveCapture(folder, rgb, {"schema_version": np.array(1)})
            self.assertEqual(existing.read_bytes(), b"keep existing image")
            self.assertEqual(list(image_folder.iterdir()), [existing])
            self.assertEqual(list((Path(folder) / "labels").iterdir()), [])

    def test_CaptureRejectsInvalidImageDimensions(self):
        camera = pv.Camera()
        light = pv.Light()
        for image_size in ((0, 120), (160, 0), (-1, 120), (160,)):
            with self.subTest(image_size=image_size), self.assertRaises(ValueError):
                CaptureFrame({}, camera, light, {}, image_size=image_size)


if __name__ == "__main__":
    unittest.main()
