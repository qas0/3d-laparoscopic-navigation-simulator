from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
# registers the render backend before rebuilding actors in a fresh process
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkProperty


def SnapshotScene(actors, camera, light, serializable=False):
    snapshot = {}
    textures = {}
    if isinstance(camera, dict):
        # rebuilds render objects only inside the capture process
        for name, sources in actors.items():
            snapshot[name] = []
            for source in sources:
                mesh = source["mesh"]
                if isinstance(mesh, dict):
                    data = mesh
                    mesh = pv.PolyData(data["points"], faces=data["faces"], lines=data["lines"],
                                       verts=data["verts"], strips=data["strips"], force_float=False)
                    for association, arrays in data["data"].items():
                        for key, value in arrays.items():
                            getattr(mesh, association)[key] = value
                    for attributes, active in zip((mesh.GetPointData(), mesh.GetCellData()), data["active"]):
                        for role, key in enumerate(active):
                            attributes.SetActiveAttribute(-1 if key is None else key, role)
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(mesh)
                for setting, value in source["mapper"].items():
                    getattr(mapper, f"Set{setting}")(value)
                if source["lookup"] is not None:
                    lookup = vtkLookupTable()
                    lookup.SetNumberOfTableValues(len(source["lookup"]["colours"]))
                    for setting, value in source["lookup"]["settings"].items():
                        getattr(lookup, f"Set{setting}")(value)
                    for index, colour in enumerate(source["lookup"]["colours"]):
                        lookup.SetTableValue(index, colour)
                    mapper.SetLookupTable(lookup)
                prop = vtkProperty()
                for setting, value in source["material"].items():
                    getattr(prop, f"Set{setting}")(value)
                for slot, data in source["textures"].items():
                    if data["key"] not in textures:
                        texture = pv.Texture(data["image"])
                        for setting, value in data["settings"].items():
                            getattr(texture, f"Set{setting}")(value)
                        textures[data["key"]] = texture
                    prop.SetTexture(slot, textures[data["key"]])
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.SetProperty(prop)
                matrix = vtkMatrix4x4()
                matrix.DeepCopy(source["matrix"].ravel())
                actor.SetUserMatrix(matrix)
                actor.SetVisibility(source["visible"])
                snapshot[name].append(actor)
        capture_camera = pv.Camera()
        for setting, value in camera["settings"].items():
            if isinstance(value, tuple):
                getattr(capture_camera, f"Set{setting}")(*value)
            else:
                getattr(capture_camera, f"Set{setting}")(value)
        capture_camera.SetModelTransformMatrix(camera["model_transform"].ravel())
        capture_light = pv.Light(light_type="scene light")
        for setting, value in light["settings"].items():
            getattr(capture_light, f"Set{setting}")(value)
        if light["transform"] is not None:
            matrix = vtkMatrix4x4()
            matrix.DeepCopy(light["transform"].ravel())
            capture_light.SetTransformMatrix(matrix)
        return snapshot, capture_camera, capture_light

    for name, sources in actors.items():
        snapshot[name] = []
        for source in sources:
            mesh = pv.wrap(source.GetMapper().GetInput())
            if serializable:
                # plain arrays avoid VTK serialization holding the UI's Python lock
                mesh = {
                    "points": np.array(mesh.points),
                    **{key: np.array(getattr(mesh, key)) for key in ("faces", "lines", "verts", "strips")},
                    "data": {association: {key: np.array(value) for key, value in getattr(mesh, association).items()}
                             for association in ("point_data", "cell_data", "field_data")},
                    "active": [[None if attributes.GetAbstractAttribute(role) is None
                                else attributes.GetAbstractAttribute(role).GetName()
                                for role in range(attributes.NUM_ATTRIBUTES)]
                               for attributes in (mesh.GetPointData(), mesh.GetCellData())]}
            else:
                mesh = mesh.copy(deep=True)
            mapper = {setting: getattr(source.GetMapper(), f"Get{setting}")()
                      for setting in ("ScalarVisibility", "ScalarMode", "ColorMode", "ArrayName", "ArrayId",
                                      "ArrayAccessMode", "ArrayComponent", "ScalarRange",
                                      "InterpolateScalarsBeforeMapping", "UseLookupTableScalarRange")}
            table = source.GetMapper().GetLookupTable()
            lookup = None
            if isinstance(table, vtkLookupTable):
                lookup = {
                    "colours": np.array([table.GetTableValue(index)
                                         for index in range(table.GetNumberOfTableValues())]),
                    "settings": {setting: getattr(table, f"Get{setting}")()
                                 for setting in ("TableRange", "Scale", "Ramp", "NanColor", "BelowRangeColor",
                                                 "AboveRangeColor", "UseBelowRangeColor", "UseAboveRangeColor")}}
            # copies material settings explicitly; VTK DeepCopy mishandles texture ownership
            material = {setting: getattr(source.GetProperty(), f"Get{setting}")()
                        for setting in ("Interpolation", "Representation", "Color", "AmbientColor", "DiffuseColor",
                                        "SpecularColor", "Ambient", "Diffuse", "Specular", "SpecularPower", "Opacity",
                                        "Metallic", "Roughness", "CoatStrength", "CoatRoughness", "CoatColor", "CoatIOR",
                                        "BaseIOR", "Anisotropy", "AnisotropyRotation", "NormalScale", "OcclusionStrength",
                                        "EmissiveFactor", "Lighting", "BackfaceCulling", "FrontfaceCulling",
                                        "ShowTexturesOnBackface", "CoatNormalScale", "EdgeVisibility", "EdgeColor",
                                        "LineWidth", "PointSize", "RenderPointsAsSpheres", "RenderLinesAsTubes")}
            material_textures = {}
            for slot in ("albedoTex", "normalTex", "materialTex", "emissiveTex", "anisotropyTex", "coatNormalTex"):
                texture = source.GetProperty().GetTexture(slot)
                if texture is None:
                    continue
                key = texture.GetAddressAsString("")
                if key not in textures:
                    textures[key] = {
                        "key": key, "image": (pv.Texture(pv.wrap(texture.GetInput())).to_array().copy()
                                               if serializable else pv.wrap(texture.GetInput()).copy(deep=True)),
                        "settings": {setting: getattr(texture, f"Get{setting}")()
                                     for setting in ("Interpolate", "Repeat", "Mipmap", "EdgeClamp", "UseSRGBColorSpace",
                                                     "Quality", "BlendingMode", "ColorMode")}}
                material_textures[slot] = textures[key]
            matrix = source.GetMatrix()
            snapshot[name].append({
                "mesh": mesh, "mapper": mapper, "lookup": lookup, "material": material,
                "textures": material_textures, "visible": source.GetVisibility(),
                "matrix": np.array([[matrix.GetElement(row, column) for column in range(4)]
                                    for row in range(4)])})
    matrix = camera.GetModelTransformMatrix()
    capture_camera = {
        "settings": {setting: getattr(camera, f"Get{setting}")()
                     for setting in ("Position", "FocalPoint", "ViewUp", "ViewAngle", "ClippingRange",
                                     "ParallelProjection", "ParallelScale", "WindowCenter", "UseHorizontalViewAngle",
                                     "ViewShear", "EyeAngle")},
        "model_transform": np.array([[matrix.GetElement(row, column) for column in range(4)]
                                     for row in range(4)])}
    matrix = light.GetTransformMatrix()
    capture_light = {
        "settings": {setting: getattr(light, f"Get{setting}")()
                     for setting in ("LightType", "Position", "FocalPoint", "AmbientColor", "DiffuseColor",
                                     "SpecularColor", "Intensity", "Switch", "Positional", "Exponent", "ConeAngle",
                                     "AttenuationValues", "ShadowAttenuation")},
        "transform": None if matrix is None else np.array([
            [matrix.GetElement(row, column) for column in range(4)] for row in range(4)])}
    if serializable:
        return snapshot, capture_camera, capture_light
    return SnapshotScene(snapshot, capture_camera, capture_light)


def CaptureFrame(actors, camera, light, metadata, image_size=(640, 480), colour_overrides=None):
    size = np.asarray(image_size)
    if (size.shape != (2,) or not np.issubdtype(size.dtype, np.integer)
            or np.any(size < 2) or np.any(size > 4096)):
        raise ValueError("Invalid image size.")
    parallel = camera["settings"]["ParallelProjection"] if isinstance(camera, dict) else camera.GetParallelProjection()
    if parallel or not 1 <= len(actors) <= 255 or "background" in actors:
        raise ValueError("Invalid capture scene.")
    labels = {name: np.asarray(value).copy() for name, value in metadata.items()}
    if any(value.dtype.hasobject for value in labels.values()):
        raise ValueError("Invalid label data.")
    actors, capture_camera, light = SnapshotScene(actors, camera, light)
    width, height = map(int, size)
    plotter = pv.Plotter(off_screen=True, window_size=(width, height), lighting="none")
    try:
        plotter.set_background("#291419", top="#4b272b")
        clones = []
        # renders only physical snapshot actors, excluding navigation overlays
        for class_id, (name, sources) in enumerate(actors.items(), 1):
            for actor in sources:
                mesh = pv.wrap(actor.GetMapper().GetInput())
                if colour_overrides is not None and name in colour_overrides:
                    actor.GetMapper().ScalarVisibilityOff()
                    actor.GetProperty().SetColor(*colour_overrides[name])
                plotter.renderer.AddActor(actor)
                clones.append((class_id, actor, mesh))
        clipping_range = capture_camera.clipping_range
        plotter.camera = capture_camera
        capture_camera.clipping_range = clipping_range
        plotter.add_light(light)
        plotter.enable_anti_aliasing("ssaa")
        rgb = plotter.screenshot(return_img=True, transparent_background=False).copy()

        # renders exact visible classes at native resolution, without blended edge IDs
        plotter.disable_anti_aliasing()
        plotter.render_window.SetMultiSamples(0)
        plotter.set_background("black")
        plotter.renderer.GradientBackgroundOff()
        for class_id, actor, mesh in clones:
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.ScalarVisibilityOff()
            prop = vtkProperty()
            prop.SetColor(class_id / 255, 0, 0)
            prop.LightingOff()
            prop.SetInterpolationToFlat()
            actor.SetMapper(mapper)
            actor.SetProperty(prop)
        plotter.render()
        mask = plotter.screenshot(return_img=True, transparent_background=False)[:, :, 0].copy()
        # preserves the camera's clipping range when reading positive optical-axis depth
        depth = -plotter.get_image_depth(fill_value=np.nan, reset_camera_clipping_range=False)
        model_view = capture_camera.GetModelViewTransformMatrix()
        world_to_camera = np.diag([1.0, -1.0, -1.0, 1.0]) @ np.array([
            [model_view.GetElement(row, column) for column in range(4)] for row in range(4)])
        projection = capture_camera.GetProjectionTransformMatrix(width / height, -1, 1)
        intrinsics = np.array([
            [projection.GetElement(0, 0) * width / 2, 0,
             (1 - projection.GetElement(0, 2)) * width / 2 - 0.5],
            [0, projection.GetElement(1, 1) * height / 2,
             (1 + projection.GetElement(1, 2)) * height / 2 - 0.5],
            [0, 0, 1]])
        labels.update(
            schema_version=np.array(1), class_ids=np.arange(len(actors) + 1, dtype=np.uint8),
            class_names=np.array(["background", *actors]), image_size=size.copy(),
            semantic_mask=mask, depth_mm=depth.copy(), camera_intrinsics=intrinsics,
            world_to_camera=world_to_camera, camera_clipping_mm=np.array(capture_camera.clipping_range),
            pixel_convention=np.array("top-left origin; integer pixel centres; u right, v down"),
            depth_convention=np.array("positive camera Z in mm; NaN background; native pixel centres"),
            rgb_antialiasing=np.array("ssaa; class + depth edges use native pixel centres"))

        if "target_world_mm" in labels:
            target = np.asarray(labels["target_world_mm"], dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("Invalid target.")
            target_camera = (world_to_camera @ np.r_[target, 1])[:3]
            uv = (intrinsics @ target_camera)[:2] / target_camera[2] if target_camera[2] > 0 else np.full(2, np.nan)
            near, far = capture_camera.clipping_range
            in_frame = bool(near <= target_camera[2] <= far and
                            np.all(uv >= -0.5) and np.all(uv < size - 0.5))
            visible = in_frame
            eye = np.asarray(capture_camera.position)
            distance = np.linalg.norm(target - eye)
            if visible and distance > 0.001:
                # stops just before the target surface so it cannot occlude itself
                start = eye + (near / target_camera[2]) * (target - eye)
                endpoint = target - 0.001 * (target - eye) / distance
                for _, actor, mesh in clones:
                    if not actor.GetVisibility():
                        continue
                    matrix = actor.GetMatrix()
                    matrix = np.array([[matrix.GetElement(row, column) for column in range(4)]
                                       for row in range(4)])
                    local = np.linalg.solve(matrix, np.column_stack((np.r_[start, 1], np.r_[endpoint, 1])))
                    hits, _ = mesh.ray_trace(local[:3, 0], local[:3, 1], first_point=True)
                    if len(hits):
                        visible = False
                        break
            labels.update(target_pixel_uv=uv, target_camera_z_mm=np.array(target_camera[2]),
                          target_in_frame=np.array(in_frame), target_visible=np.array(visible))
        return rgb, labels
    finally:
        plotter.close()


def SaveCapture(folder, rgb, labels):
    rgb = np.asarray(rgb)
    labels = {name: np.asarray(value) for name, value in labels.items()}
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Invalid capture image.")
    if any(value.dtype.hasobject for value in labels.values()):
        raise ValueError("Invalid label data.")
    image_folder, label_folder = Path(folder) / "images", Path(folder) / "labels"
    image_folder.mkdir(parents=True, exist_ok=True)
    label_folder.mkdir(parents=True, exist_ok=True)
    indices = [int(path.stem[6:]) for directory, suffix in ((image_folder, ".png"), (label_folder, ".npz"))
               for path in directory.glob(f"frame_*{suffix}") if path.stem[6:].isdigit()]
    index = max(indices, default=0) + 1
    while True:
        image_path = image_folder / f"frame_{index:06d}.png"
        label_path = label_folder / f"frame_{index:06d}.npz"
        try:
            image_file = image_path.open("xb")
        except FileExistsError:
            index += 1
            continue
        try:
            label_file = label_path.open("xb")
        except FileExistsError:
            image_file.close()
            image_path.unlink()
            index += 1
            continue
        except BaseException:
            image_file.close()
            image_path.unlink()
            raise
        try:
            with image_file, label_file:
                Image.fromarray(rgb).save(image_file, format="PNG")
                np.savez_compressed(label_file, **labels)
        except BaseException:
            image_path.unlink()
            label_path.unlink()
            raise
        return image_path, label_path
