#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _parse_cli_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    ap = argparse.ArgumentParser(description="Headless Blender shadow render for mesh vertex sequences.")
    ap.add_argument("--vertices_npy", type=str, required=True)
    ap.add_argument("--faces_npy", type=str, required=True)
    ap.add_argument("--frames_dir", type=str, required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument(
        "--quality_profile",
        type=str,
        default="standard",
        choices=["standard", "paper"],
        help="paper enables cleaner orthographic lighting/material defaults.",
    )
    ap.add_argument("--render_device", type=str, default="cuda", choices=["cpu", "cuda", "optix"])
    ap.add_argument("--camera_offset", type=str, required=True, help="Comma-separated camera offset in Blender coords.")
    ap.add_argument(
        "--yaw_deg",
        type=float,
        default=0.0,
        help="Rotate the actor around the vertical Blender Z axis before rendering.",
    )
    ap.add_argument("--camera_mode", type=str, default="track", choices=["track", "fixed"])
    ap.add_argument("--camera_projection", type=str, default="perspective", choices=["perspective", "orthographic"])
    ap.add_argument("--ortho_scale", type=float, default=0.0, help="0 = auto fit actor height.")
    ap.add_argument(
        "--fixed_anchor",
        type=str,
        default="first",
        choices=["first", "mean", "origin"],
        help="Reference point used when camera_mode=fixed.",
    )
    ap.add_argument("--sun_energy", type=float, default=2.7)
    ap.add_argument("--sun_elev_deg", type=float, default=42.0)
    ap.add_argument("--sun_azim_deg", type=float, default=35.0)
    ap.add_argument("--sun_angle_deg", type=float, default=7.0)
    ap.add_argument("--area_light_energy", type=float, default=0.0)
    ap.add_argument("--area_light_size", type=float, default=4.0)
    ap.add_argument("--area_light_offset", type=str, default="0.0,-2.8,3.2")
    ap.add_argument("--floor_mode", type=str, default="plane", choices=["plane", "none"])
    ap.add_argument("--plane_color", type=str, default="0.93,0.93,0.93,1.0")
    ap.add_argument("--mesh_color", type=str, default="0.78,0.78,0.80,1.0")
    ap.add_argument("--mesh_roughness", type=float, default=0.55)
    ap.add_argument("--mesh_subsurface", type=float, default=0.0)
    ap.add_argument("--world_strength", type=float, default=0.85)
    ap.add_argument(
        "--white_background",
        action="store_true",
        help="Use Standard color management with pure white world background.",
    )
    ap.add_argument(
        "--transparent_background",
        action="store_true",
        help="Render PNG frames with transparent film background.",
    )
    return ap.parse_args(argv)


def _parse_floats(text: str, expected: int) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(text).split(","))
    if len(vals) != expected:
        raise ValueError(f"Expected {expected} comma-separated floats, got {text}")
    return vals


def _rotate_vertices_yaw(vertices: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    if abs(yaw) < 1e-6:
        return vertices
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float32,
    )
    center_xy = vertices[..., :2].mean(axis=(0, 1), keepdims=True)
    xy = vertices[..., :2] - center_xy
    rotated = vertices.copy()
    rotated[..., :2] = np.einsum("...i,ij->...j", xy, rot.T) + center_xy
    return rotated


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def _setup_cycles(
    width: int,
    height: int,
    samples: int,
    render_device: str,
    world_strength: float,
    white_background: bool,
    transparent_background: bool,
    *,
    cycles_gpu_device_index: int | None = None,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent_background else "RGB"
    scene.render.film_transparent = bool(transparent_background)
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transparent_max_bounces = 4
    if white_background or transparent_background:
        try:
            scene.view_settings.view_transform = "Standard"
        except Exception:
            pass
        try:
            scene.view_settings.look = "None"
        except Exception:
            pass
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    else:
        scene.view_settings.exposure = -0.35
        try:
            scene.view_settings.look = "Medium High Contrast"
        except Exception:
            pass

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputWorld")
    if white_background:
        # Keep the camera background pure white while preserving softer world lighting.
        light_path = nodes.new(type="ShaderNodeLightPath")
        bg_lighting = nodes.new(type="ShaderNodeBackground")
        bg_camera = nodes.new(type="ShaderNodeBackground")
        mix = nodes.new(type="ShaderNodeMixShader")
        bg_lighting.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_lighting.inputs[1].default_value = float(world_strength)
        bg_camera.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_camera.inputs[1].default_value = 1.0
        links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
        links.new(bg_lighting.outputs["Background"], mix.inputs[1])
        links.new(bg_camera.outputs["Background"], mix.inputs[2])
        links.new(mix.outputs["Shader"], output.inputs["Surface"])
    else:
        bg = nodes.new(type="ShaderNodeBackground")
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg.inputs[1].default_value = float(world_strength)
        links.new(bg.outputs["Background"], output.inputs["Surface"])

    scene.cycles.device = "CPU"
    if render_device.lower() == "cpu":
        return

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX" if render_device.lower() == "optix" else "CUDA"
        prefs.get_devices()
        rd = render_device.lower().strip()

        if cycles_gpu_device_index is not None and int(cycles_gpu_device_index) >= 0:
            want = int(cycles_gpu_device_index)
            if rd == "optix":
                candidates = [d for d in prefs.devices if getattr(d, "type", "") == "OPTIX"]
            else:
                candidates = [d for d in prefs.devices if getattr(d, "type", "") == "CUDA"]
            for dev in prefs.devices:
                if getattr(dev, "type", "") in {"CUDA", "OPTIX"}:
                    dev.use = 0
            enabled = 0
            if want < len(candidates):
                candidates[want].use = 1
                enabled = 1
                scene.cycles.device = "GPU"
                print(f"[INFO] Blender Cycles using single GPU index {want}: {candidates[want].name}")
            elif candidates:
                candidates[0].use = 1
                enabled = 1
                scene.cycles.device = "GPU"
                print(
                    f"[WARN] cycles_gpu_device_index={want} out of range ({len(candidates)} device(s)); "
                    f"using index 0: {candidates[0].name}"
                )
            else:
                print("[WARN] No CUDA/OPTIX devices found; fallback to CPU")
        else:
            enabled = 0
            for dev in prefs.devices:
                dev.use = 0
                if rd == "optix":
                    if dev.type == "OPTIX":
                        dev.use = 1
                        enabled += 1
                elif dev.type in {"CUDA", "OPTIX"}:
                    dev.use = 1
                    enabled += 1
            if enabled > 0:
                scene.cycles.device = "GPU"
                print(f"[INFO] Blender Cycles using GPU devices: {enabled}")
            else:
                print("[WARN] No compatible GPU device found for Cycles; fallback to CPU")
    except Exception as exc:  # pragma: no cover - blender runtime specific
        print(f"[WARN] Failed to enable GPU Cycles ({exc}); fallback to CPU")


def _set_input_if_present(bsdf: bpy.types.Node, names: tuple[str, ...], value) -> None:
    for name in names:
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value
            return


def _make_principled_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    subsurface: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    _set_input_if_present(bsdf, ("Roughness",), float(roughness))
    if float(subsurface) > 0.0:
        _set_input_if_present(bsdf, ("Subsurface Weight", "Subsurface"), float(subsurface))
        _set_input_if_present(bsdf, ("Subsurface Color",), color)
    return mat


def _add_floor(center_xy: tuple[float, float], plane_size: float, color: tuple[float, float, float, float]) -> None:
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(float(center_xy[0]), float(center_xy[1]), 0.0))
    plane = bpy.context.active_object
    plane.name = "Floor"
    plane.scale = (float(plane_size) * 0.5, float(plane_size) * 0.5, 1.0)
    plane.data.materials.append(_make_principled_material("FloorMat", color=color, roughness=0.85))


def _add_sun(sun_energy: float, sun_elev_deg: float, sun_azim_deg: float, sun_angle_deg: float) -> None:
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 0.0))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = float(sun_energy)
    sun.data.angle = math.radians(float(sun_angle_deg))
    sun.rotation_euler = (
        math.radians(90.0 - float(sun_elev_deg)),
        0.0,
        math.radians(float(sun_azim_deg)),
    )


def _add_area_light(
    center_xy: tuple[float, float],
    area_light_offset: np.ndarray,
    energy: float,
    size: float,
) -> None:
    if float(energy) <= 0.0:
        return
    bpy.ops.object.light_add(
        type="AREA",
        location=(
            float(center_xy[0] + area_light_offset[0]),
            float(center_xy[1] + area_light_offset[1]),
            float(area_light_offset[2]),
        ),
    )
    light = bpy.context.active_object
    light.name = "Softbox"
    light.data.energy = float(energy)
    light.data.size = float(size)


def _setup_camera(
    camera_offset: np.ndarray,
    camera_projection: str,
    ortho_scale: float,
    perspective_fov_deg: float = 45.0,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    cam_data = bpy.data.cameras.new("Camera")
    if str(camera_projection).strip().lower() == "orthographic":
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = float(ortho_scale)
    else:
        cam_data.lens_unit = "FOV"
        cam_data.angle = math.radians(float(perspective_fov_deg))
    camera = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    target = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(target)

    track = camera.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    camera.location = Vector(camera_offset.tolist())
    return camera, target


def _camera_anchor(centers: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).strip().lower()
    if mode == "origin":
        return np.zeros(3, dtype=np.float32)
    if mode == "mean":
        return centers.mean(axis=0).astype(np.float32)
    return centers[0].astype(np.float32)


def _create_mesh_object(
    vertices0: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float, float],
    roughness: float,
    subsurface: float,
) -> tuple[bpy.types.Object, bpy.types.Mesh]:
    mesh = bpy.data.meshes.new("ActorMesh")
    mesh.from_pydata(vertices0.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("Actor", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(
        _make_principled_material(
            "ActorMat",
            color=color,
            roughness=roughness,
            subsurface=subsurface,
        )
    )
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj, mesh


def _update_mesh_vertices(mesh: bpy.types.Mesh, vertices: np.ndarray) -> None:
    mesh.vertices.foreach_set("co", vertices.reshape(-1).tolist())
    mesh.update()


def main() -> None:
    args = _parse_cli_args()
    vertices = np.load(str(Path(args.vertices_npy).expanduser().resolve()))
    faces = np.load(str(Path(args.faces_npy).expanduser().resolve()))
    frames_dir = Path(args.frames_dir).expanduser().resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)

    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Expected vertices shape (T, V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"Expected faces shape (F, 3), got {faces.shape}")
    if args.quality_profile == "paper":
        args.samples = max(int(args.samples), 96)
        args.camera_projection = "orthographic"
        args.sun_energy = float(args.sun_energy) if args.sun_energy != 2.7 else 2.1
        args.sun_angle_deg = max(float(args.sun_angle_deg), 10.0)
        args.area_light_energy = max(float(args.area_light_energy), 260.0)
        args.world_strength = max(float(args.world_strength), 0.55)
        args.mesh_roughness = max(float(args.mesh_roughness), 0.68)
        args.mesh_subsurface = max(float(args.mesh_subsurface), 0.08)
    vertices = _rotate_vertices_yaw(vertices.astype(np.float32, copy=False), args.yaw_deg)

    camera_offset = np.asarray(_parse_floats(args.camera_offset, 3), dtype=np.float32)
    area_light_offset = np.asarray(_parse_floats(args.area_light_offset, 3), dtype=np.float32)
    plane_color = _parse_floats(args.plane_color, 4)
    mesh_color = _parse_floats(args.mesh_color, 4)
    centers = vertices.mean(axis=1)
    fixed_anchor = _camera_anchor(centers, args.fixed_anchor)
    center_xy = (float(np.mean(centers[:, 0])), float(np.mean(centers[:, 1])))
    plane_size = max(6.0, float(max(np.ptp(vertices[..., 0]), np.ptp(vertices[..., 1])) * 2.6))
    bbox_z = float(np.ptp(vertices[..., 2]))
    ortho_scale = float(args.ortho_scale) if float(args.ortho_scale) > 0.0 else max(2.15, bbox_z * 1.28)

    _clear_scene()
    _setup_cycles(
        args.width,
        args.height,
        args.samples,
        args.render_device,
        args.world_strength,
        args.white_background,
        args.transparent_background,
    )
    if args.floor_mode == "plane":
        _add_floor(center_xy, plane_size, plane_color)
    _add_sun(args.sun_energy, args.sun_elev_deg, args.sun_azim_deg, args.sun_angle_deg)
    _add_area_light(center_xy, area_light_offset, args.area_light_energy, args.area_light_size)
    camera, target = _setup_camera(camera_offset, args.camera_projection, ortho_scale)
    _obj, mesh = _create_mesh_object(
        vertices[0],
        faces.astype(np.int32),
        mesh_color,
        args.mesh_roughness,
        args.mesh_subsurface,
    )

    scene = bpy.context.scene
    for frame_idx in range(vertices.shape[0]):
        verts = vertices[frame_idx].astype(np.float32, copy=False)
        _update_mesh_vertices(mesh, verts)
        center = centers[frame_idx].astype(np.float32, copy=False)
        if args.camera_mode == "fixed":
            target.location = Vector(fixed_anchor.tolist())
            camera.location = Vector((fixed_anchor + camera_offset).tolist())
        else:
            target.location = Vector(center.tolist())
            camera.location = Vector((center + camera_offset).tolist())
        bpy.context.view_layer.update()
        scene.render.filepath = str(frames_dir / f"frame_{frame_idx:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"[render] {frame_idx + 1}/{vertices.shape[0]}")


if __name__ == "__main__":
    main()
