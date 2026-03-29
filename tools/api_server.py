"""
VistaDream API Server

FastAPI server that exposes VistaDream pipeline steps as REST endpoints,
enabling SuperSplat editor to drive the pipeline interactively.

Usage:
    pixi run python tools/api_server.py
    # or with uvicorn directly:
    uvicorn tools.api_server:app --host 0.0.0.0 --port 7860
"""

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="VistaDream API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session state ────────────────────────────────────────────────────────────

SessionStatus = Literal["idle", "running", "done", "error"]


@dataclass
class Session:
    session_id: str
    session_dir: Path
    status: SessionStatus = "idle"
    stage: str = ""
    message: str = ""
    coarse_ply: Path | None = None
    refined_ply: Path | None = None
    pipeline: object = None  # StepwisePipeline, lazy import
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()

SESSIONS_BASE = Path("data/api_sessions")


def _get_session(session_id: str) -> Session:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return session


# ─── Request / Response models ────────────────────────────────────────────────


class CoarseRequest(BaseModel):
    n_frames: int = 8
    max_resolution: int = 512
    expansion_percent: float = 0.3
    num_steps: int = 25
    guidance: float = 30.0
    use_quantized_flux: bool = False


class CameraPose(BaseModel):
    position: dict  # {x, y, z}
    target: dict    # {x, y, z}


class RefineRequest(BaseModel):
    cameras: list[CameraPose]
    n_train_iters: int = 500


# ─── Background task helpers ──────────────────────────────────────────────────


def _run_coarse(session: Session, image_path: Path, req: CoarseRequest) -> None:
    from vistadream.api.stepwise_pipeline import StepwisePipeline

    pipeline = StepwisePipeline()

    def _progress(msg: str) -> None:
        with session.lock:
            session.message = msg

    try:
        with session.lock:
            session.status = "running"
            session.stage = "coarse"
            session.message = "Initializing pipeline..."

        pipeline.initialize(
            image_path=image_path,
            session_dir=session.session_dir,
            n_frames=req.n_frames,
            max_resolution=req.max_resolution,
            expansion_percent=req.expansion_percent,
            num_steps=req.num_steps,
            guidance=req.guidance,
            use_quantized_flux=req.use_quantized_flux,
        )

        ply_path = pipeline.run_coarse(progress_cb=_progress)

        with session.lock:
            session.pipeline = pipeline
            session.coarse_ply = ply_path
            session.status = "done"
            session.message = "Coarse stage complete."

    except Exception as exc:
        with session.lock:
            session.status = "error"
            session.message = f"Error: {exc}"
        raise


def _run_refine(session: Session, req: RefineRequest) -> None:
    def _progress(msg: str) -> None:
        with session.lock:
            session.message = msg

    try:
        with session.lock:
            pipeline = session.pipeline
            session.status = "running"
            session.stage = "refine"
            session.message = "Starting user-guided refine..."

        if pipeline is None:
            raise RuntimeError("Pipeline not initialized. Run coarse stage first.")

        camera_poses = [
            {"position": cam.position, "target": cam.target}
            for cam in req.cameras
        ]

        ply_path = pipeline.run_user_refine(
            camera_poses=camera_poses,
            n_train_iters=req.n_train_iters,
            progress_cb=_progress,
        )

        with session.lock:
            session.refined_ply = ply_path
            session.status = "done"
            session.message = "Refine complete."

    except Exception as exc:
        with session.lock:
            session.status = "error"
            session.message = f"Error: {exc}"
        raise


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/sessions")
def create_session() -> dict:
    """Create a new pipeline session. Returns session_id."""
    session_id = str(uuid.uuid4())
    session_dir = SESSIONS_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    session = Session(session_id=session_id, session_dir=session_dir)
    with _sessions_lock:
        _sessions[session_id] = session
    return {"session_id": session_id}


@app.post("/sessions/{session_id}/image")
async def upload_image(session_id: str, file: UploadFile) -> dict:
    """Upload the input image for this session."""
    session = _get_session(session_id)
    suffix = Path(file.filename).suffix or ".jpg"
    image_path = session.session_dir / f"input{suffix}"
    contents = await file.read()
    image_path.write_bytes(contents)
    with session.lock:
        session.status = "idle"
        session.message = f"Image uploaded: {file.filename}"
    return {"image_path": str(image_path), "size": len(contents)}


@app.post("/sessions/{session_id}/coarse")
def start_coarse(
    session_id: str,
    req: CoarseRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Start coarse pipeline stage in background. Poll /status for progress."""
    session = _get_session(session_id)

    with session.lock:
        if session.status == "running":
            return {"detail": "Already running"}

    # Find the uploaded image
    image_files = list(session.session_dir.glob("input.*"))
    if not image_files:
        raise HTTPException(status_code=400, detail="Upload an image first (/image endpoint)")

    image_path = image_files[0]
    background_tasks.add_task(_run_coarse, session, image_path, req)
    return {"detail": "Coarse stage started"}


@app.get("/sessions/{session_id}/status")
def get_status(session_id: str) -> dict:
    """Poll processing status."""
    session = _get_session(session_id)
    with session.lock:
        return {
            "status": session.status,
            "stage": session.stage,
            "message": session.message,
            "has_coarse": session.coarse_ply is not None and session.coarse_ply.exists(),
            "has_refined": session.refined_ply is not None and session.refined_ply.exists(),
        }


@app.get("/sessions/{session_id}/ply/{stage}")
def download_ply(session_id: str, stage: Literal["coarse", "refined"]) -> FileResponse:
    """Download PLY file for the given stage ('coarse' or 'refined')."""
    session = _get_session(session_id)
    with session.lock:
        ply_path = session.coarse_ply if stage == "coarse" else session.refined_ply

    if ply_path is None or not ply_path.exists():
        raise HTTPException(status_code=404, detail=f"No {stage} PLY available yet")

    return FileResponse(
        path=str(ply_path),
        media_type="application/octet-stream",
        filename=f"vistadream_{stage}.ply",
    )


@app.post("/sessions/{session_id}/refine")
def start_refine(
    session_id: str,
    req: RefineRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Start user-guided refine with the provided camera poses."""
    session = _get_session(session_id)

    with session.lock:
        if session.status == "running":
            return {"detail": "Already running"}
        if session.pipeline is None:
            raise HTTPException(status_code=400, detail="Run coarse stage first")

    if not req.cameras:
        raise HTTPException(status_code=400, detail="Provide at least one camera pose")

    background_tasks.add_task(_run_refine, session, req)
    return {"detail": f"Refine started with {len(req.cameras)} camera(s)"}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """Clean up a session and its temp files."""
    session = _get_session(session_id)
    import shutil
    shutil.rmtree(session.session_dir, ignore_errors=True)
    with _sessions_lock:
        del _sessions[session_id]
    return {"detail": "Session deleted"}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VistaDream API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(Path(__file__).parent),
    )
