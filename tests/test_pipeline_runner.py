"""Unit and integration tests for app.pipeline.runner.

Verifies:
1. Standalone Pipeline Runner without Celery / Database dependencies.
2. PipelineMetrics and PipelineResult behavior and serialization.
3. Modular execution: run_timesync, run_reconstruction, run_materials, and end-to-end run().
4. Zero dependency on Celery / SQLAlchemy in app/pipeline and app/services.
"""

import ast
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import pytest

from app.pipeline.runner import (
    PipelineMetrics,
    PipelineResult,
    PipelineRunner,
    collect_pipeline_artifacts,
    extract_pipeline_metrics,
    run_3d_reconstruction,
    run_material_estimation,
    run_pipeline,
    run_sensor_ingestion,
)


def test_pipeline_metrics_defaults_and_serialization():
    metrics = PipelineMetrics(
        room_area_m2=15.5,
        room_perimeter_m=16.0,
        room_height_m=2.6,
        room_width_m=3.5,
        room_depth_m=4.43,
        wall_count=4,
        portal_count=1,
    )
    d = metrics.to_dict()
    assert d["room_area_m2"] == 15.5
    assert d["wall_count"] == 4
    assert d["portal_count"] == 1
    assert isinstance(d["materials"], dict)
    assert isinstance(d["timings"], dict)


def test_pipeline_result_dataclass_and_dict_access():
    metrics = PipelineMetrics(room_area_m2=20.0)
    result = PipelineResult(
        success=True,
        session_dir="/tmp/test_session",
        metrics=metrics,
        artifacts={"floorplan.json": "/tmp/test_session/floorplan.json"},
        floorplan={"room": {"area_m2": 20.0}},
    )
    assert result.success is True
    assert result["success"] is True
    assert "success" in result
    assert "non_existent" not in result
    assert result.get("success") is True
    assert result.get("non_existent", "fallback") == "fallback"
    assert "success" in result.keys()
    assert result["session_dir"] == "/tmp/test_session"
    assert result["metrics"].room_area_m2 == 20.0

    assert "room_area_m2" in metrics
    assert "missing_key" not in metrics
    assert metrics["room_area_m2"] == 20.0
    assert metrics.get("room_area_m2") == 20.0
    assert metrics.get("missing_key", 999) == 999
    assert "room_area_m2" in metrics.keys()

    d = result.to_dict()
    assert d["success"] is True
    assert d["metrics"]["room_area_m2"] == 20.0
    assert "floorplan.json" in d["artifacts"]


def test_pipeline_runner_zero_celery_and_zero_db_imports():
    """Verify app/pipeline/ and app/services/ have zero Celery and zero SQLAlchemy imports."""
    root_dir = Path(__file__).resolve().parent.parent
    checked_dirs = [root_dir / "app" / "pipeline", root_dir / "app" / "services"]
    forbidden_modules = {"celery", "sqlalchemy"}

    for cdir in checked_dirs:
        assert cdir.is_dir(), f"Directory {cdir} does not exist"
        for py_path in cdir.rglob("*.py"):
            with open(py_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        assert top not in forbidden_modules, (
                            f"Forbidden import '{alias.name}' in {py_path}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        assert top not in forbidden_modules, (
                            f"Forbidden import from '{node.module}' in {py_path}"
                        )
                        # Also assert no imports from app.tasks, app.models, app.core.celery_app
                        assert not node.module.startswith("app.tasks"), (
                            f"Forbidden import from '{node.module}' in {py_path}"
                        )
                        assert not node.module.startswith("app.models"), (
                            f"Forbidden import from '{node.module}' in {py_path}"
                        )
                        assert not node.module.startswith("app.core.celery_app"), (
                            f"Forbidden import from '{node.module}' in {py_path}"
                        )


def test_collect_pipeline_artifacts(tmp_path):
    (tmp_path / "reconstructed.ply").write_text("ply")
    (tmp_path / "reconstructed.glb").write_bytes(b"glb")
    (tmp_path / "floorplan.json").write_text("{}")
    (tmp_path / "A3_Combined.png").write_bytes(b"png")
    (tmp_path / "A4_Floorplan.pdf").write_bytes(b"pdf")
    (tmp_path / "Wall_Elevations.png").write_bytes(b"png")
    (tmp_path / "unrelated.txt").write_text("ignore")

    artifacts = collect_pipeline_artifacts(str(tmp_path))
    assert "reconstructed.ply" in artifacts
    assert "reconstructed.glb" in artifacts
    assert "floorplan.json" in artifacts
    assert "A3_Combined.png" in artifacts
    assert "A4_Floorplan.pdf" in artifacts
    assert "Wall_Elevations.png" in artifacts
    assert "unrelated.txt" not in artifacts


def test_extract_pipeline_metrics_from_floorplan(tmp_path):
    fp_data = {
        "room": {
            "area_m2": 12.34,
            "perimeter_m": 14.2,
            "height_meters": 2.4,
            "width_m": 3.1,
            "depth_m": 3.98,
        },
        "walls": [{"wall_index": 0}, {"wall_index": 1}],
        "portals": [{"portal_type": "door"}],
    }
    (tmp_path / "floorplan.json").write_text(json.dumps(fp_data))
    (tmp_path / "materials.json").write_text(json.dumps({"flooring_m2": 12.34}))

    metrics = extract_pipeline_metrics(str(tmp_path))
    assert metrics.room_area_m2 == pytest.approx(12.34)
    assert metrics.room_perimeter_m == pytest.approx(14.2)
    assert metrics.room_height_m == pytest.approx(2.4)
    assert metrics.room_width_m == pytest.approx(3.1)
    assert metrics.room_depth_m == pytest.approx(3.98)
    assert metrics.wall_count == 2
    assert metrics.portal_count == 1
    assert metrics.materials.get("flooring_m2") == pytest.approx(12.34)


def test_pipeline_runner_handles_failure_gracefully(tmp_path):
    runner = PipelineRunner()
    # Non-existent session dir should return PipelineResult with success=False
    non_existent = str(tmp_path / "non_existent_dir_12345")
    res = runner.run(non_existent)
    assert res.success is False
    assert res.error_message is not None


def test_standalone_pipeline_runner_end_to_end_mocked(tmp_path, monkeypatch):
    """Test full PipelineRunner.run() execution without Celery or DB."""
    session_dir = tmp_path / "session_test"
    session_dir.mkdir()

    # Create minimum viable files
    (session_dir / "manifest.json").write_text('{"camera_intrinsics": [[50,0,32],[0,50,24],[0,0,1]]}')
    (session_dir / "camera_matrix.csv").write_text("50.0,0.0,32.0\n0.0,50.0,24.0\n0.0,0.0,1.0\n")
    (session_dir / "odometry.csv").write_text("timestamp,frame,x,y,z,qx,qy,qz,qw\n0.0,0,0,0,0,0,0,0,1\n")
    (session_dir / "processed_vio.csv").write_text("device_timestamp_ns,x,y,z,qx,qy,qz,qw\n0,0,0,0,0,0,0,1\n")
    (session_dir / "processed_lidar.csv").write_text("device_timestamp_ns,x,y,depth\n0,10,10,2.5\n")

    rgb_dir = session_dir / "rgb"
    rgb_dir.mkdir()
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    (rgb_dir / "frame_0.jpg").write_bytes(buf.tobytes())

    # Mock DepthCompletionService
    class MockDepth:
        def run_depthor_plus(self, rgb, sparse, k, **kwargs):
            h, w = rgb.shape[:2]
            return np.full((h, w), 2.0, dtype=np.float32)

    monkeypatch.setattr("app.services.depth_completion.DepthCompletionService", MockDepth)

    # Mock segment_planes
    rect_layout = {
        "vertices": {"v0": [0.0, 0.0], "v1": [3.0, 0.0], "v2": [3.0, 4.0], "v3": [0.0, 4.0]},
        "walls": [
            {"wall_index": 0, "joints": ["v0", "v1"], "thickness_meters": 0.15, "height_meters": 2.5},
            {"wall_index": 1, "joints": ["v1", "v2"], "thickness_meters": 0.15, "height_meters": 2.5},
            {"wall_index": 2, "joints": ["v2", "v3"], "thickness_meters": 0.15, "height_meters": 2.5},
            {"wall_index": 3, "joints": ["v3", "v0"], "thickness_meters": 0.15, "height_meters": 2.5},
        ],
        "portals": [],
        "floor": None,
        "ceiling": None,
        "gravity_vector": [0.0, 0.0, -9.8],
    }
    monkeypatch.setattr(
        "app.services.reconstruction.ReconstructionService.segment_planes",
        lambda self, mesh, g, **kwargs: (rect_layout, o3d.geometry.PointCloud()),
    )

    runner = PipelineRunner(enable_server_vio=True, enable_tof_pipeline=True)
    result = runner.run(str(session_dir))

    assert result.success is True
    assert result.error_message is None
    assert (session_dir / "reconstructed.ply").is_file()
    assert (session_dir / "reconstructed.glb").is_file()
    assert (session_dir / "floorplan.json").is_file()
    assert (session_dir / "floorplan.svg").is_file()
    assert (session_dir / "floorplan.dxf").is_file()
    assert (session_dir / "room_model.glb").is_file()
    assert result.metrics.room_area_m2 == pytest.approx(12.0)
    assert result.metrics.wall_count == 4


def test_pipeline_runner_skips_ingestion_when_already_ingested(tmp_path):
    session_dir = tmp_path / "preprocessed_session"
    session_dir.mkdir()
    (session_dir / "processed_lidar.csv").write_text("device_timestamp_ns,x,y,depth\n0,1,1,2.0\n10,1,1,2.0\n")
    (session_dir / "imu.csv").write_text("raw,imu,data\n")
    runner = PipelineRunner()
    res = runner.run(str(session_dir))
    # Should not raise "LiDAR telemetry must contain at least 2 rows"
    assert "LiDAR telemetry must contain at least 2 rows" not in (res.error_message or "")


try:
    from app.tasks.pipeline import run_pipeline_task
    HAS_CELERY_TASKS = True
except ImportError:
    HAS_CELERY_TASKS = False


@pytest.mark.skipif(not HAS_CELERY_TASKS, reason="Celery tasks not present in serverless repo")
@pytest.mark.asyncio
async def test_run_pipeline_task_celery_db_success(db_session, tmp_path, monkeypatch):
    """Verify run_pipeline_task updates SQLite DB session to completed with metrics and artifacts."""
    import uuid
    from app.core.config import settings
    from app.models.session import Session
    from app.tasks.pipeline import run_pipeline_task

    session_id = str(uuid.uuid4())
    session_dir = tmp_path / session_id
    session_dir.mkdir()
    (session_dir / "floorplan.json").write_text('{"room": {"area_m2": 18.5}}')

    monkeypatch.setattr(settings, "SESSION_DIR", str(tmp_path))
    sess = Session(id=session_id, user_id="test_user", status="received")
    db_session.add(sess)
    await db_session.commit()

    mock_runner = Mock()
    mock_runner.recalculate_params = {"rec": True}
    mock_runner.run.return_value = PipelineResult(
        success=True,
        session_dir=str(session_dir),
        metrics=PipelineMetrics(room_area_m2=18.5, wall_count=4),
        artifacts={"floorplan.json": str(session_dir / "floorplan.json")},
        floorplan={"room": {"area_m2": 18.5}},
    )

    with patch("app.tasks.pipeline.PipelineRunner", return_value=mock_runner):
        task_res = run_pipeline_task.apply(args=[session_id]).get()

    assert task_res["success"] is True

    await db_session.refresh(sess)
    assert sess.status == "completed"
    assert sess.progress_percentage == 100
    assert sess.results is not None
    assert sess.results["room_area_m2"] == pytest.approx(18.5)
    assert "floorplan.json" in sess.results["artifacts"]


@pytest.mark.skipif(not HAS_CELERY_TASKS, reason="Celery tasks not present in serverless repo")
@pytest.mark.asyncio
async def test_run_pipeline_task_celery_db_failure(db_session, tmp_path, monkeypatch):
    """Verify run_pipeline_task transitions SQLite DB session to failed on runner error."""
    import uuid
    from app.core.config import settings
    from app.models.session import Session
    from app.tasks.pipeline import run_pipeline_task

    session_id = str(uuid.uuid4())
    session_dir = tmp_path / session_id
    session_dir.mkdir()

    monkeypatch.setattr(settings, "SESSION_DIR", str(tmp_path))
    sess = Session(id=session_id, user_id="test_user", status="received")
    db_session.add(sess)
    await db_session.commit()

    mock_runner = Mock()
    mock_runner.recalculate_params = {}
    mock_runner.run.return_value = PipelineResult(
        success=False,
        session_dir=str(session_dir),
        metrics=PipelineMetrics(),
        artifacts={},
        error_message="Severe TSDF Empty Mesh Error",
    )

    with patch("app.tasks.pipeline.PipelineRunner", return_value=mock_runner):
        with pytest.raises(RuntimeError, match="Severe TSDF Empty Mesh Error"):
            run_pipeline_task.apply(args=[session_id]).get()

    await db_session.refresh(sess)
    assert sess.status == "failed"
    assert "Severe TSDF Empty Mesh Error" in (sess.error_message or "")


@pytest.mark.skipif(not HAS_CELERY_TASKS, reason="Celery tasks not present in serverless repo")
def test_run_pipeline_task_signature():
    """Verify run_pipeline_task accepts single session_id string parameter."""
    import inspect
    from app.tasks.pipeline import run_pipeline_task
    sig = inspect.signature(run_pipeline_task.run)
    params = list(sig.parameters.keys())
    assert "session_id" in params

