"""
Step-wise VistaDream pipeline for interactive use via the API server.

Exposes coarse and user-guided refine stages independently,
so SuperSplat can drive the pipeline step by step.
"""

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from jaxtyping import Float

from vistadream.api.single_img_pipeline import SingleImageConfig, SingleImagePipeline, pose_to_frame
from vistadream.ops.gs.basic import Frame, Gaussian_Scene, save_ply
from vistadream.ops.gs.train import GS_Train_Tool
from simplecv.rerun_log_utils import RerunTyroConfig


def _lookat_cam_T_world(
    pos: Float[np.ndarray, "3"],
    target: Float[np.ndarray, "3"],
) -> Float[np.ndarray, "4 4"]:
    """Build cam_T_world from position and target (VistaDream Y-down world).

    Follows the same convention as Traj_Base.rot_by_look_at with camera_up=[0,-1,0].
    """
    direction = target - pos
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(4, dtype=np.float64)
    direction = direction / norm

    # VistaDream uses camera_up=[0,-1,0] which gets negated → effective_up=[0,1,0]
    effective_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(effective_up, direction)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        # degenerate: forward is parallel to up, pick a different up
        effective_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(effective_up, direction)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    up_final = np.cross(direction, right)

    # world_T_cam: columns are camera axes in world space
    world_T_cam = np.eye(4, dtype=np.float64)
    world_T_cam[:3, :3] = np.column_stack([right, up_final, direction])
    world_T_cam[:3, 3] = pos

    cam_T_world = np.linalg.inv(world_T_cam)
    return cam_T_world.astype(np.float32)


def supersplat_pose_to_cam_T_world(
    position: dict,
    target: dict,
) -> Float[np.ndarray, "4 4"]:
    """Convert SuperSplat camera pose (Y-up world) to VistaDream cam_T_world (Y-down world).

    SuperSplat uses PlayCanvas Y-up coordinate system.
    VistaDream uses Y-down world (first camera is identity, RDF convention).
    Conversion: flip Y component of position and target.
    """
    # Flip Y axis to convert from Y-up (PlayCanvas) to Y-down (VistaDream)
    pos_vd = np.array([position["x"], -position["y"], position["z"]], dtype=np.float64)
    tgt_vd = np.array([target["x"], -target["y"], target["z"]], dtype=np.float64)
    return _lookat_cam_T_world(pos_vd, tgt_vd)


class StepwisePipeline:
    """Interactive step-wise wrapper around SingleImagePipeline.

    Usage:
        pipeline = StepwisePipeline()
        pipeline.initialize(image_path, stage="coarse", ...)
        ply_path = pipeline.run_coarse(progress_cb)
        ply_path = pipeline.run_user_refine(camera_poses, progress_cb)
    """

    def __init__(self) -> None:
        self._pipeline: SingleImagePipeline | None = None
        self._save_dir: Path = Path("data/api_sessions")

    def initialize(
        self,
        image_path: Path,
        session_dir: Path,
        n_frames: int = 8,
        max_resolution: int = 512,
        expansion_percent: float = 0.3,
        num_steps: int = 25,
        guidance: float = 30.0,
        use_quantized_flux: bool = False,
    ) -> None:
        """Initialize pipeline (loads models). This is the slow step."""
        rr_config = RerunTyroConfig(application_id="vistadream-api")

        config = SingleImageConfig(
            rr_config=rr_config,
            image_path=image_path,
            stage="coarse",
            disable_rerun=True,
            n_frames=n_frames,
            max_resolution=max_resolution,
            expansion_percent=expansion_percent,
            num_steps=num_steps,
            guidance=guidance,
            use_quantized_flux=use_quantized_flux,
        )
        self._pipeline = SingleImagePipeline(config)
        self._session_dir = session_dir
        self._session_dir.mkdir(parents=True, exist_ok=True)

    def run_coarse(self, progress_cb: Callable[[str], None] | None = None) -> Path:
        """Run coarse pipeline stage and return path to saved PLY."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        def _log(msg: str) -> None:
            print(f"[coarse] {msg}")
            if progress_cb:
                progress_cb(msg)

        _log("Setting up...")
        self._pipeline.setup_rerun()

        _log("Initializing scene (depth estimation + outpainting)...")
        self._pipeline._initialize()

        _log("Generating coarse frames (Flux inpainting)...")
        self._pipeline._coarse()

        ply_path = self._session_dir / "coarse.ply"
        _log(f"Saving coarse PLY to {ply_path}...")
        save_ply(self._pipeline.scene, ply_path)

        _log("Coarse stage complete.")
        return ply_path

    def run_user_refine(
        self,
        camera_poses: list[dict],
        n_train_iters: int = 500,
        progress_cb: Callable[[str], None] | None = None,
    ) -> Path:
        """Add user-specified camera views, inpaint, retrain and save refined PLY.

        Args:
            camera_poses: List of {position: {x,y,z}, target: {x,y,z}} in SuperSplat coords.
            n_train_iters: Number of Gaussian training iterations after adding new frames.
            progress_cb: Optional callback for progress messages.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        if not self._pipeline.scene.frames:
            raise RuntimeError("Coarse stage must be run before refine.")

        def _log(msg: str) -> None:
            print(f"[refine] {msg}")
            if progress_cb:
                progress_cb(msg)

        _log(f"Adding {len(camera_poses)} user camera views...")

        for i, cam_pose in enumerate(camera_poses):
            _log(f"Processing camera {i + 1}/{len(camera_poses)}...")

            cam_T_world = supersplat_pose_to_cam_T_world(
                cam_pose["position"], cam_pose["target"]
            )

            # Generate frame at this camera position using the existing scene
            frame: Frame = pose_to_frame(self._pipeline.scene, cam_T_world, margin=32)

            # Inpaint the frame using Flux
            inpainted_frame: Frame = self._pipeline._inpaint_next_frame(frame)

            # Add to scene
            self._pipeline.scene._add_trainable_frame(inpainted_frame, require_grad=True)

        _log(f"Training Gaussians for {n_train_iters} iterations...")
        self._pipeline.scene = GS_Train_Tool(
            self._pipeline.scene, iters=n_train_iters
        )(self._pipeline.scene.frames, log=False)

        ply_path = self._session_dir / "refined.ply"
        _log(f"Saving refined PLY to {ply_path}...")
        save_ply(self._pipeline.scene, ply_path)

        _log("User refine complete.")
        return ply_path
