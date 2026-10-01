import json
import os
import shutil
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

import boto3
import gdown
import numpy as np
import open3d as o3d
import requests
import torch
from botocore.config import Config
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


APP_NAME = "VGGT Room3D Extended Backend"
BASE_DIR = os.getenv("BASE_DIR", "/app/vggt_room3d_jobs")
MODEL_ID = "facebook/VGGT-1B-Commercial"
DEFAULT_HARD_MAX_POINTS = 300_000_000
R2_BUCKET = os.getenv("R2_BUCKET", "3d-ply")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL") or (f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else "")
R2_PRESIGN_EXPIRES = int(os.getenv("R2_PRESIGN_EXPIRES", str(3 * 24 * 60 * 60)))
DEFAULT_CLEAN_VOXEL = 0.012
DEFAULT_CLEAN_STAT_NEIGHBORS = 16
DEFAULT_CLEAN_STAT_STD = 2.8

os.makedirs(BASE_DIR, exist_ok=True)
app = FastAPI(title=APP_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    dtype = torch.float32

model = None


def get_model():
    global model
    if model is None:
        print(f"Loading VGGT model: {MODEL_ID}")
        print("Device:", device)
        print("Dtype:", dtype)
        model = VGGT.from_pretrained(MODEL_ID).to(device)
        model.eval()
        print("VGGT loaded.")
    return model


def is_image(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))


def safe_name(filename: str) -> str:
    return os.path.basename(filename).replace(" ", "_")


def safe_object_part(value: str) -> str:
    value = str(value or "item")
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in value).strip("_")
    return cleaned or "item"


def clean_stem(path: Path) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem).strip("_")
    return cleaned or "point_cloud"


def get_r2_client():
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    if not R2_ENDPOINT_URL or not access_key or not secret_key:
        return None
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def upload_to_r2(local_path: str, key: str) -> dict:
    client = get_r2_client()
    if client is None:
        print(f"[WARN] R2 credentials not configured. Skipping remote upload for {key}. Keeping local: {local_path}", flush=True)
        return {
            "bucket": None,
            "key": key,
            "size": int(os.path.getsize(local_path)) if os.path.isfile(local_path) else 0,
            "etag": "",
            "presigned_url": None,
            "local_path": local_path,
        }
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=f,
            ContentType="application/octet-stream",
        )
    head = client.head_object(Bucket=R2_BUCKET, Key=key)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": key},
        ExpiresIn=R2_PRESIGN_EXPIRES,
    )
    return {
        "bucket": R2_BUCKET,
        "key": key,
        "size": int(head.get("ContentLength", os.path.getsize(local_path))),
        "etag": str(head.get("ETag", "")).strip('\"'),
        "presigned_url": url,
        "expires_in": R2_PRESIGN_EXPIRES,
    }


def clean_ply_file(path: str, output_dir: str, voxel: float, stat_neighbors: int, stat_std: float) -> dict:
    input_path = Path(path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty():
        raise RuntimeError(f"Cannot read point cloud or file has no points: {input_path}")

    original_points = len(pcd.points)
    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    after_voxel_points = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=stat_neighbors, std_ratio=stat_std)

    output_path = out_dir / f"{clean_stem(input_path)}_clean.ply"
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False, compressed=False)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "mode": "ultra_fast",
        "original_points": original_points,
        "after_voxel_points": after_voxel_points,
        "output_points": len(pcd.points),
        "settings": {"voxel": voxel, "stat_neighbors": stat_neighbors, "stat_std": stat_std},
    }
    report_path = out_dir / f"{clean_stem(input_path)}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source) + 1e-9
    target /= np.linalg.norm(target) + 1e-9

    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-9:
        return np.eye(3)

    vx = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (cross_norm ** 2))


def angle_to_axis(vector: np.ndarray) -> tuple[int, float]:
    vector = vector / (np.linalg.norm(vector) + 1e-9)
    axis = int(np.argmax(np.abs(vector)))
    angle = np.degrees(np.arccos(np.clip(abs(vector[axis]), -1.0, 1.0)))
    return axis, float(angle)


def fit_plane_svd(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    return normal / (np.linalg.norm(normal) + 1e-9)


def detect_floor_normal(points: np.ndarray, max_sample: int = 220_000):
    rng = np.random.default_rng(7)
    sample = points
    if len(sample) > max_sample:
        sample = sample[rng.choice(len(sample), max_sample, replace=False)]

    diagonal = float(np.linalg.norm(np.ptp(sample, axis=0)))
    threshold = max(diagonal * 0.008, 0.012)
    best = None

    # VGGT does not guarantee which world axis is vertical. Test the low band
    # of every axis and keep the strongest plane aligned with that axis.
    for up_axis in range(3):
        values = sample[:, up_axis]
        low = np.percentile(values, 2)
        high = np.percentile(values, 25)
        candidates = sample[(values >= low) & (values <= high)]
        if len(candidates) < 1000:
            continue

        best_plane = None
        for _ in range(1800):
            ids = rng.choice(len(candidates), 3, replace=False)
            a, b, c = candidates[ids]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue

            normal /= norm
            distances = np.abs((candidates - a) @ normal)
            inlier_mask = distances < threshold
            count = int(inlier_mask.sum())
            if best_plane is None or count > best_plane[0]:
                best_plane = (count, inlier_mask)

        if best_plane is None:
            continue

        count, inlier_mask = best_plane
        refined = fit_plane_svd(candidates[inlier_mask])
        if refined[up_axis] < 0:
            refined = -refined
        _, angle = angle_to_axis(refined)
        score = count / max(angle + 1.0, 1.0)
        item = (score, up_axis, refined, count, len(candidates), angle, threshold)
        if best is None or item[0] > best[0]:
            best = item

    if best is None:
        raise RuntimeError("Could not detect a floor-like plane for Manhattan alignment")

    _, up_axis, normal, inliers, candidates, angle, threshold = best
    return up_axis, normal, inliers, candidates, angle, threshold


def yaw_rotation_matrix(up_axis: int, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    if up_axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if up_axis == 1:
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def detect_manhattan_yaw(points: np.ndarray, up_axis: int, max_sample: int = 220_000) -> float:
    rng = np.random.default_rng(11)
    sample = points
    if len(sample) > max_sample:
        sample = sample[rng.choice(len(sample), max_sample, replace=False)]

    axes = [axis for axis in range(3) if axis != up_axis]
    footprint = sample[:, axes]
    best_angle = 0.0
    best_area = float("inf")

    for angle in np.linspace(-45.0, 45.0, 361):
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        rotated = footprint @ np.array([[c, -s], [s, c]]).T
        extent = rotated.max(axis=0) - rotated.min(axis=0)
        area = float(extent[0] * extent[1])
        if area < best_area:
            best_area = area
            best_angle = float(angle)

    return best_angle


def align_clean_ply_to_manhattan(path: str) -> dict:
    clean_path = Path(path)
    pcd = o3d.io.read_point_cloud(str(clean_path))
    if pcd.is_empty():
        raise RuntimeError(f"Cannot read clean point cloud: {clean_path}")

    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    up_axis, normal, inliers, candidates, floor_tilt, threshold = detect_floor_normal(points)

    target = np.zeros(3)
    target[up_axis] = 1.0
    level_rotation = rotation_from_vectors(normal, target)
    leveled = (points - center) @ level_rotation.T + center

    yaw_deg = detect_manhattan_yaw(leveled, up_axis)
    yaw_rotation = yaw_rotation_matrix(up_axis, yaw_deg)
    aligned = (leveled - center) @ yaw_rotation.T + center

    # Canonical output for 3D viewers: X/Y floor plane and Z as height.
    z_up_rotation = rotation_from_vectors(target, np.array([0.0, 0.0, 1.0]))
    aligned = (aligned - center) @ z_up_rotation.T + center
    floor_z = float(np.percentile(aligned[:, 2], 2))
    aligned[:, 2] -= floor_z

    # Overwrite the clean file so there is only one final PLY artifact.
    pcd.points = o3d.utility.Vector3dVector(aligned)
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        combined_rotation = z_up_rotation @ yaw_rotation @ level_rotation
        pcd.normals = o3d.utility.Vector3dVector(normals @ combined_rotation.T)
    if not o3d.io.write_point_cloud(str(clean_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f"Could not write Manhattan-aligned point cloud: {clean_path}")

    info = {
        "applied": True,
        "method": "floor_ransac_svd_and_min_area_yaw",
        "output": str(clean_path),
        "input_up_axis": "XYZ"[up_axis],
        "output_up_axis": "Z",
        "floor_normal_before": normal.tolist(),
        "floor_tilt_deg": float(floor_tilt),
        "floor_inliers": int(inliers),
        "floor_candidates": int(candidates),
        "plane_threshold": float(threshold),
        "yaw_correction_deg": float(yaw_deg),
        "z_floor_offset": floor_z,
    }
    report_path = clean_path.with_name(f"{clean_path.stem}_alignment.json")
    report_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    info["report_path"] = str(report_path)
    return info


def zip_job(job_dir: str, job_id: str) -> str:
    zip_path = os.path.join(job_dir, f"{job_id}_result.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(job_dir):
            for file in files:
                full_path = os.path.join(root, file)
                if full_path == zip_path:
                    continue
                z.write(full_path, os.path.relpath(full_path, job_dir))
    return zip_path


def tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return x


def write_binary_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        vertex.tofile(f)


def export_point_cloud(predictions, image_names: list[str], out_dir: str, conf_threshold: float, max_points: int, hard_max_points: int) -> dict:
    max_points = min(int(max_points), int(hard_max_points), DEFAULT_HARD_MAX_POINTS)
    world_points = tensor_to_numpy(predictions["world_points"])
    world_conf = tensor_to_numpy(predictions["world_points_conf"])
    images = tensor_to_numpy(predictions["images"])

    pts_all, rgb_all, conf_all = [], [], []
    per_view = {}
    for view_idx in range(world_points.shape[1]):
        pts = world_points[0, view_idx]
        conf = world_conf[0, view_idx]
        img = np.transpose(images[0, view_idx], (1, 2, 0))
        img = ((img + 1.0) * 127.5) if img.min() < 0 else (img * 255.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        mask = np.isfinite(pts).all(axis=-1) & np.isfinite(conf) & (conf > conf_threshold)
        name = image_names[view_idx] if view_idx < len(image_names) else f"view_{view_idx:03d}.jpg"
        safe_key = name.replace(".", "_").replace("/", "_").replace(" ", "_")
        per_view[f"view_points__{safe_key}"] = pts.astype(np.float32)
        per_view[f"view_conf__{safe_key}"] = conf.astype(np.float32)
        per_view[f"view_colors__{safe_key}"] = img.astype(np.uint8)
        if np.any(mask):
            pts_all.append(pts[mask])
            rgb_all.append(img[mask])
            conf_all.append(conf[mask])

    if not pts_all:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
        conf = np.empty((0,), dtype=np.float32)
    else:
        points = np.concatenate(pts_all, axis=0).astype(np.float32)
        colors = np.concatenate(rgb_all, axis=0).astype(np.uint8)
        conf = np.concatenate(conf_all, axis=0).astype(np.float32)

    if len(points) > max_points:
        idx = np.argsort(conf)[-max_points:]
        points, colors, conf = points[idx], colors[idx], conf[idx]

    npz_path = os.path.join(out_dir, "project_point_cloud.npz")
    ply_path = os.path.join(out_dir, "project_point_cloud.ply")
    np.savez_compressed(
        npz_path,
        points=points,
        colors=colors,
        confidence=conf,
        image_names=np.array(image_names),
        **per_view,
    )
    write_binary_ply(ply_path, points, colors)
    return {"num_points": int(len(points)), "npz_path": npz_path, "ply_path": ply_path, "max_points": int(max_points)}


def prediction_shapes(predictions) -> dict:
    shapes = {}
    for key, value in predictions.items():
        if hasattr(value, "shape"):
            shapes[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            shapes[key] = str(type(value))
    return shapes


@app.get("/")
def home():
    return {"status": "ok", "app": APP_NAME, "device": device, "dtype": str(dtype), "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model_loaded": model is not None}


def download_and_extract_zip(url: str, job_dir: str, dest_dir: str) -> List[str]:
    zip_path = os.path.join(job_dir, "temp_images.zip")
    
    # 1. Download the file
    print(f"Downloading ZIP from: {url}")
    if "drive.google.com" in url or "docs.google.com" in url:
        gdown.download(url, zip_path, quiet=False, fuzzy=True)
    else:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        raise RuntimeError(f"Failed to download zip file from URL: {url}")
        
    # 2. Extract the file
    print(f"Extracting ZIP to: {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
        
    # Clean up the downloaded ZIP
    try:
        os.remove(zip_path)
    except OSError:
        pass
        
    # 3. Find image paths
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")
    image_paths = []
    
    for root, _, files in os.walk(dest_dir):
        for file in files:
            full_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(full_path)
            
    return sorted(image_paths)


def find_multisensor_session_dir(root_dir: str) -> Optional[str]:
    """Detect if directory contains multi-sensor LiDAR/odometry scan files."""
    indicators = ("manifest.json", "camera_matrix.csv", "odometry.csv", "lidar.csv")
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f in filenames for f in indicators):
            return dirpath
    return None


def run_multisensor_pipeline(session_dir: str, batch_id: str, job_id: str, out_dir: str) -> dict:
    from app.pipeline.runner import run_pipeline
    from app.services.cad.contract import layout_to_metrics_mm

    result = run_pipeline(session_dir)
    if not result.success:
        raise RuntimeError(f"Multi-sensor pipeline execution failed: {result.error_message}")

    object_prefix = f"{safe_object_part(batch_id)}/{job_id}"

    clean_ply_candidates = [
        os.path.join(session_dir, "reconstructed_visual.ply"),
        os.path.join(session_dir, "whiteflat.ply"),
        os.path.join(session_dir, "reconstructed.ply"),
    ]
    clean_ply_path = next((p for p in clean_ply_candidates if os.path.isfile(p)), None)
    clean_ply_r2 = {}
    if clean_ply_path:
        clean_ply_r2 = upload_to_r2(
            clean_ply_path, f"ply_clean/{object_prefix}/project_point_cloud_clean.ply"
        )

    floorplan_files = {}
    file_candidates = [
        ("floorplan_png", ["floorplan.png", "ISO_A3_Floorplan.png", "ISO_A4_Floorplan.png"]),
        ("floorplan_pdf", ["floorplan.pdf", "ISO_A3_Floorplan.pdf", "ISO_A4_Floorplan.pdf"]),
        ("wall_elevation_png", ["Walls.png", "walls.png"]),
        ("debug_topdown_png", ["debug_topdown.png"]),
        ("room_model_glb", ["room_model.glb", "room_model_texture.glb", "reconstructed.glb"]),
        ("metrics_json", ["metrics.json"]),
    ]
    for file_key, cands in file_candidates:
        for c in cands:
            p = os.path.join(session_dir, c)
            if os.path.isfile(p):
                target_name = "walls.png" if "wall" in file_key else c
                floorplan_files[file_key] = upload_to_r2(
                    p, f"ply_clean/{object_prefix}/{target_name}"
                )
                break

    room_metrics = None
    metrics_json_path = os.path.join(session_dir, "metrics.json")
    if os.path.isfile(metrics_json_path):
        try:
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                room_metrics = json.load(f)
        except Exception:
            room_metrics = None
    if room_metrics is None and result.floorplan:
        room_metrics = layout_to_metrics_mm(result.floorplan)

    presigned_url = clean_ply_r2.get("presigned_url") or floorplan_files.get("floorplan_png", {}).get("presigned_url")
    r2_key = clean_ply_r2.get("key")

    meta = {
        "status": "success",
        "job_id": job_id,
        "batch_id": batch_id,
        "mode": "multisensor_lidar",
        "download_url": presigned_url,
        "clean_ply": {
            "presigned_url": presigned_url,
            "r2_key": r2_key,
        },
        "room_metrics": room_metrics,
        "floorplan_files": floorplan_files,
        "timings": result.metrics.timings if hasattr(result.metrics, "timings") else {},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def run_inference_pipeline(
    image_paths: List[str],
    batch_id: str,
    metadata: list,
    conf_threshold: float,
    max_points: int,
    hard_max_points: int,
    max_images: Optional[int],
    clean_voxel: float,
    clean_stat_neighbors: int,
    clean_stat_std: float,
    job_id: str,
    job_dir: str,
    img_dir: str,
    out_dir: str
) -> dict:
    image_paths = sorted(image_paths)
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]
    if not image_paths:
        raise RuntimeError("No valid images found for processing")

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    m = get_model()
    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = m(images)
        else:
            predictions = m(images)

    torch.save(predictions, os.path.join(out_dir, "predictions.pt"))
    image_names = [os.path.basename(p) for p in image_paths]
    cloud_info = export_point_cloud(predictions, image_names, out_dir, conf_threshold, max_points, hard_max_points)
    clean_dir = os.path.join(out_dir, "clean")
    clean_info = clean_ply_file(
        cloud_info["ply_path"],
        clean_dir,
        clean_voxel,
        clean_stat_neighbors,
        clean_stat_std,
    )

    alignment_info = align_clean_ply_to_manhattan(clean_info["output"])

    object_prefix = f"{safe_object_part(batch_id)}/{job_id}"
    clean_r2 = upload_to_r2(clean_info["output"], f"ply_clean/{object_prefix}/project_point_cloud_clean.ply")
    try:
        os.remove(cloud_info["ply_path"])
    except OSError:
        pass

    # ── 3D to 2D Floorplan & Room Metrics Extraction ──
    room_metrics = None
    floorplan_files = {}
    try:
        from floorplan_generator.main import generate_floorplan_from_ply
        fp_result = generate_floorplan_from_ply(
            pointcloud_path=clean_info["output"],
            output_dir=clean_dir,
            project_name=safe_object_part(batch_id),
            sheet_size="A3",
            no_wall_slice=True,
            projection_view="xy_pos_z",
            alignment_mode="off",
        )
        room_metrics = fp_result.get("metrics")
        fp_files = fp_result.get("files", {})

        # Upload 2D outputs to R2 alongside 3D PLY if files exist
        if fp_files.get("floorplan_png") and os.path.exists(fp_files["floorplan_png"]):
            floorplan_files["floorplan_png"] = upload_to_r2(
                fp_files["floorplan_png"], f"ply_clean/{object_prefix}/floorplan.png"
            )
        if fp_files.get("floorplan_pdf") and os.path.exists(fp_files["floorplan_pdf"]):
            floorplan_files["floorplan_pdf"] = upload_to_r2(
                fp_files["floorplan_pdf"], f"ply_clean/{object_prefix}/floorplan.pdf"
            )
        if fp_files.get("wall_elevation_png") and os.path.exists(fp_files["wall_elevation_png"]):
            floorplan_files["wall_elevation_png"] = upload_to_r2(
                fp_files["wall_elevation_png"], f"ply_clean/{object_prefix}/walls.png"
            )
        if fp_files.get("debug_topdown_png") and os.path.exists(fp_files["debug_topdown_png"]):
            floorplan_files["debug_topdown_png"] = upload_to_r2(
                fp_files["debug_topdown_png"], f"ply_clean/{object_prefix}/debug_topdown.png"
            )
        if fp_files.get("metrics_json") and os.path.exists(fp_files["metrics_json"]):
            floorplan_files["metrics_json"] = upload_to_r2(
                fp_files["metrics_json"], f"ply_clean/{object_prefix}/metrics.json"
            )
    except Exception as exc:
        print(f"Warning: Failed to extract 2D floorplan metrics on AI server: {exc}")
        traceback.print_exc()

    meta = {
        "status": "success",
        "job_id": job_id,
        "batch_id": batch_id,
        "room_metrics": room_metrics,
        "floorplan_files": floorplan_files,
        "num_images": len(image_paths),
        "image_names": image_names,
        "device": device,
        "dtype": str(dtype),
        "model_id": MODEL_ID,
        "conf_threshold": float(conf_threshold),
        "point_cloud": {
            "num_points": cloud_info["num_points"],
            "max_points": cloud_info["max_points"],
            "files": ["output/clean/project_point_cloud_clean.ply"],
        },
        "cleaning": {
            "mode": clean_info["mode"],
            "original_points": clean_info["original_points"],
            "after_voxel_points": clean_info["after_voxel_points"],
            "output_points": clean_info["output_points"],
            "settings": clean_info["settings"],
        },
        "alignment": alignment_info,
        "storage": {
            "provider": "cloudflare_r2",
            "bucket": R2_BUCKET,
            "endpoint": R2_ENDPOINT_URL,
            "expires_in": R2_PRESIGN_EXPIRES,
            "expires_days": round(R2_PRESIGN_EXPIRES / 86400, 2),
        },
        "clean_ply": clean_r2,
        "prediction_keys": list(predictions.keys()),
        "shape_info": prediction_shapes(predictions),
        "download_url": f"/download/{job_id}",
        "npz_url": f"/download_npz/{job_id}",
        "job_url": f"/jobs/{job_id}",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    zip_job(job_dir, job_id)
    if device == "cuda":
        torch.cuda.empty_cache()
    return meta


@app.post("/predict_project_batch")
async def predict_project_batch(
    files: List[UploadFile] = File(...),
    batch_id: str = Form("batch"),
    metadata_json: str = Form("[]"),
    conf_threshold: float = Form(1.0),
    max_points: int = Form(3_000_000),
    hard_max_points: int = Form(DEFAULT_HARD_MAX_POINTS),
    max_images: Optional[int] = Form(None),
    clean_voxel: float = Form(DEFAULT_CLEAN_VOXEL),
    clean_stat_neighbors: int = Form(DEFAULT_CLEAN_STAT_NEIGHBORS),
    clean_stat_std: float = Form(DEFAULT_CLEAN_STAT_STD),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(BASE_DIR, job_id)
    img_dir = os.path.join(job_dir, "images")
    out_dir = os.path.join(job_dir, "output")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        metadata = json.loads(metadata_json) if metadata_json else []
        image_paths = []
        for f in files:
            name = safe_name(f.filename)
            if not is_image(name):
                continue
            path = os.path.join(img_dir, name)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            image_paths.append(path)

        if not image_paths:
            return JSONResponse(status_code=400, content={"status": "error", "error": "No valid images uploaded"})

        meta = run_inference_pipeline(
            image_paths=image_paths,
            batch_id=batch_id,
            metadata=metadata,
            conf_threshold=conf_threshold,
            max_points=max_points,
            hard_max_points=hard_max_points,
            max_images=max_images,
            clean_voxel=clean_voxel,
            clean_stat_neighbors=clean_stat_neighbors,
            clean_stat_std=clean_stat_std,
            job_id=job_id,
            job_dir=job_dir,
            img_dir=img_dir,
            out_dir=out_dir
        )
        return meta
    except Exception as e:
        if device == "cuda":
            torch.cuda.empty_cache()
        error_text = traceback.format_exc()
        with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(error_text)
        zip_job(job_dir, job_id)
        return JSONResponse(status_code=500, content={"status": "error", "job_id": job_id, "error": str(e), "download_url": f"/download/{job_id}"})


class ZipUrlRequest(BaseModel):
    zip_url: str
    batch_id: str = "batch"
    metadata: list = []
    conf_threshold: float = 1.0
    max_points: int = 3000000
    hard_max_points: int = DEFAULT_HARD_MAX_POINTS
    max_images: Optional[int] = None
    clean_voxel: float = DEFAULT_CLEAN_VOXEL
    clean_stat_neighbors: int = DEFAULT_CLEAN_STAT_NEIGHBORS
    clean_stat_std: float = DEFAULT_CLEAN_STAT_STD


@app.post("/predict_zip_url")
async def predict_zip_url(req: ZipUrlRequest):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(BASE_DIR, job_id)
    img_dir = os.path.join(job_dir, "images")
    out_dir = os.path.join(job_dir, "output")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        image_paths = download_and_extract_zip(req.zip_url, job_dir, img_dir)
        session_dir = find_multisensor_session_dir(img_dir) or find_multisensor_session_dir(job_dir)
        if session_dir is not None:
            meta = run_multisensor_pipeline(session_dir, req.batch_id, job_id, out_dir)
            zip_job(job_dir, job_id)
            return meta

        if not image_paths:
            return JSONResponse(status_code=400, content={"status": "error", "error": "No valid images or sensor data found in the zip file"})

        meta = run_inference_pipeline(
            image_paths=image_paths,
            batch_id=req.batch_id,
            metadata=req.metadata,
            conf_threshold=req.conf_threshold,
            max_points=req.max_points,
            hard_max_points=req.hard_max_points,
            max_images=req.max_images,
            clean_voxel=req.clean_voxel,
            clean_stat_neighbors=req.clean_stat_neighbors,
            clean_stat_std=req.clean_stat_std,
            job_id=job_id,
            job_dir=job_dir,
            img_dir=img_dir,
            out_dir=out_dir
        )
        return meta
    except Exception as e:
        if device == "cuda":
            torch.cuda.empty_cache()
        error_text = traceback.format_exc()
        with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(error_text)
        zip_job(job_dir, job_id)
        return JSONResponse(status_code=500, content={"status": "error", "job_id": job_id, "error": str(e), "download_url": f"/download/{job_id}"})


@app.post("/predict_zip_url_form")
async def predict_zip_url_form(
    zip_url: str = Form(...),
    batch_id: str = Form("batch"),
    metadata_json: str = Form("[]"),
    conf_threshold: float = Form(1.0),
    max_points: int = Form(3_000_000),
    hard_max_points: int = Form(DEFAULT_HARD_MAX_POINTS),
    max_images: Optional[int] = Form(None),
    clean_voxel: float = Form(DEFAULT_CLEAN_VOXEL),
    clean_stat_neighbors: int = Form(DEFAULT_CLEAN_STAT_NEIGHBORS),
    clean_stat_std: float = Form(DEFAULT_CLEAN_STAT_STD),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(BASE_DIR, job_id)
    img_dir = os.path.join(job_dir, "images")
    out_dir = os.path.join(job_dir, "output")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        metadata = json.loads(metadata_json) if metadata_json else []
        image_paths = download_and_extract_zip(zip_url, job_dir, img_dir)
        session_dir = find_multisensor_session_dir(img_dir) or find_multisensor_session_dir(job_dir)
        if session_dir is not None:
            meta = run_multisensor_pipeline(session_dir, batch_id, job_id, out_dir)
            zip_job(job_dir, job_id)
            return meta

        if not image_paths:
            return JSONResponse(status_code=400, content={"status": "error", "error": "No valid images or sensor data found in the zip file"})

        meta = run_inference_pipeline(
            image_paths=image_paths,
            batch_id=batch_id,
            metadata=metadata,
            conf_threshold=conf_threshold,
            max_points=max_points,
            hard_max_points=hard_max_points,
            max_images=max_images,
            clean_voxel=clean_voxel,
            clean_stat_neighbors=clean_stat_neighbors,
            clean_stat_std=clean_stat_std,
            job_id=job_id,
            job_dir=job_dir,
            img_dir=img_dir,
            out_dir=out_dir
        )
        return meta
    except Exception as e:
        if device == "cuda":
            torch.cuda.empty_cache()
        error_text = traceback.format_exc()
        with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(error_text)
        zip_job(job_dir, job_id)
        return JSONResponse(status_code=500, content={"status": "error", "job_id": job_id, "error": str(e), "download_url": f"/download/{job_id}"})


@app.get("/download/{job_id}")
def download(job_id: str):
    zip_path = os.path.join(BASE_DIR, job_id, f"{job_id}_result.zip")
    if not os.path.exists(zip_path):
        return JSONResponse(status_code=404, content={"status": "error", "error": "Result not found"})
    return FileResponse(zip_path, filename=f"{job_id}_result.zip", media_type="application/zip")


@app.get("/download_npz/{job_id}")
def download_npz(job_id: str):
    npz_path = os.path.join(BASE_DIR, job_id, "output", "project_point_cloud.npz")
    if not os.path.exists(npz_path):
        return JSONResponse(status_code=404, content={"status": "error", "error": "NPZ not found"})
    return FileResponse(npz_path, filename=f"{job_id}_point_cloud.npz", media_type="application/octet-stream")


@app.get("/jobs/{job_id}")
def check_job(job_id: str):
    job_dir = os.path.join(BASE_DIR, job_id)
    if not os.path.exists(job_dir):
        return JSONResponse(status_code=404, content={"status": "error", "error": "Job not found"})
    files = []
    for root, _, names in os.walk(job_dir):
        for name in names:
            files.append(os.path.relpath(os.path.join(root, name), job_dir))
    return {"status": "ok", "job_id": job_id, "files": sorted(files), "download_url": f"/download/{job_id}"}
