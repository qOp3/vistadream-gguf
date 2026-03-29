"""
Fine-stage Gaussian refinement using Multi-view Consistent Score (MCS).

Ported from the original VistaDream pipe/refine_mvdps.py:
    Coarse Gaussian Rendering -- RGB-D as init
    RGB-D add noise (MV init)
    Cycling:
        denoise to x0 -- optimize Gaussian
        re-rendering RGB-D
        render RGB-D to rectified noise
        noise rectification
        step denoise with rectified noise
    -- Finally the Gaussian
"""

from copy import deepcopy

import numpy as np
import PIL.Image
import torch
import tqdm

from vistadream.ops.gs.basic import Frame, Gaussian_Scene
from vistadream.ops.gs.train import GS_Train_Tool, RGB_Loss
from vistadream.ops.trajs import _generate_trajectory
from vistadream.ops.utils import inpaint_tiny_holes


class Refinement_Tool_MCS:
    """
    Refine a coarse Gaussian scene using diffusion-guided multi-view consistency.

    The refinement alternates between:
    1. Diffusion denoising step to predict x0 (clean image estimate)
    2. Gaussian optimization to match x0 across all refinement views
    3. Re-render refined scene and rectify the diffusion noise estimate
    """

    def __init__(
        self,
        coarse_GS: Gaussian_Scene,
        device: str = "cuda",
        refiner=None,
        traj_type: str = "spiral",
        n_view: int = 8,
        rect_w: float = 0.7,
        pre_blur: bool = False,
        n_gsopt_iters: int = 256,
    ) -> None:
        self.n_view = n_view
        self.rect_w = rect_w
        self.pre_blur = pre_blur
        self.n_gsopt_iters = n_gsopt_iters
        self.coarse_GS = coarse_GS
        self.refine_frames: list[Frame] = []
        self.process_res = 512
        self.device = device
        self.traj_type = traj_type
        # diffusion model (HackSD_MCS)
        self.RGB_LCM = refiner
        self.RGB_LCM.to("cuda")
        self.steps = self.RGB_LCM.denoise_steps
        # encode text prompt for diffusion guidance
        prompt: str = self.coarse_GS.frames[-1].prompt
        self.rgb_prompt_latent = self.RGB_LCM.model._encode_text_prompt(prompt)
        # loss function
        self.rgb_lossfunc = RGB_Loss(w_ssim=0.2)

    def _pre_process(self) -> None:
        """Generate refinement camera frames along a trajectory and compute inpaint masks."""
        strict_times = 32
        origin_H: int = self.coarse_GS.frames[0].H
        origin_W: int = self.coarse_GS.frames[0].W
        self.target_H, self.target_W = self.process_res, self.process_res
        intrinsic = deepcopy(self.coarse_GS.frames[0].intrinsic)
        H_ratio = self.target_H / origin_H
        W_ratio = self.target_W / origin_W
        intrinsic[0] *= W_ratio
        intrinsic[1] *= H_ratio
        target_H = self.target_H + 2 * strict_times
        target_W = self.target_W + 2 * strict_times
        intrinsic[0, -1] = target_W / 2
        intrinsic[1, -1] = target_H / 2
        # generate camera trajectory, skip the first pose (identity/input)
        trajs = _generate_trajectory(None, self.coarse_GS, nframes=self.n_view + 1)[1:]
        for pose in trajs:
            fine_frame = Frame(
                H=target_H,
                W=target_W,
                intrinsic=deepcopy(intrinsic),
                cam_T_world=np.linalg.inv(pose),
                prompt=self.coarse_GS.frames[-1].prompt,
            )
            self.refine_frames.append(fine_frame)
        # determine inpaint mask for each refinement frame
        temp_scene = Gaussian_Scene()
        for frame in self.coarse_GS.frames:
            if frame.keep:
                temp_scene._add_trainable_frame(frame, require_grad=False)
        for frame in self.refine_frames:
            frame = temp_scene._render_for_inpaint(frame)
        del temp_scene

    def _mv_init(self) -> None:
        """Encode coarse-rendered views as diffusion initial latents."""
        rgbs = []
        for frame in self.refine_frames:
            render_rgb, render_dpt, render_alpha = self.coarse_GS._render_RGBD(frame)
            if self.pre_blur:
                render_rgb = inpaint_tiny_holes(
                    render_rgb.cpu().numpy(),
                    render_alpha.squeeze().cpu().numpy(),
                    0.9,
                )
                render_rgb = torch.from_numpy(render_rgb.astype(np.float32)).to(render_dpt)
            rgbs.append(render_rgb.permute(2, 0, 1)[None])
        self.rgbs = torch.cat(rgbs, dim=0)
        self.RGB_LCM._encode_mv_init_images(self.rgbs)

    def _to_cuda(self, tensor: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(tensor.astype(np.float32)).to("cuda")

    def _x0_rectification(self, denoise_rgb: torch.Tensor, iters: int) -> None:
        """Optimize all Gaussian parameters to match the denoised x0 prediction."""
        CGS = deepcopy(self.coarse_GS)
        for gf in CGS.gaussian_frames:
            gf._require_grad(True)
        self.refine_GS = GS_Train_Tool(CGS)
        for _ in range(iters):
            loss = torch.tensor(0.0, device="cuda")
            # supervise on original kept frames
            nkpt = 0
            for keep_frame in self.coarse_GS.frames:
                if not keep_frame.keep:
                    continue
                nkpt += 1
                render_rgb, _, _ = self.refine_GS._render(keep_frame)
                loss_rgb = self.rgb_lossfunc(
                    render_rgb, self._to_cuda(keep_frame.rgb), valid_mask=keep_frame.inpaint
                )
                loss = loss + loss_rgb * len(self.refine_frames)
            # supervise on multi-view diffusion predictions
            for i, frame in enumerate(self.refine_frames):
                render_rgb, _, _ = self.refine_GS._render(frame)
                loss_rgb_item = self.rgb_lossfunc(denoise_rgb[i], render_rgb)
                loss = loss + loss_rgb_item * nkpt / 2.0
            loss.backward()
            self.refine_GS.optimizer.step()
            self.refine_GS.optimizer.zero_grad()

    def _step_gaussian_optimization(self, step: int):
        """Denoise to x0 and optimize Gaussians to match it."""
        with torch.no_grad():
            rgb_t = self.RGB_LCM.timesteps[-self.steps + step]
            rgb_t = torch.tensor([rgb_t]).to(self.device)
            rgb_noise_pr, rgb_denoise = self.RGB_LCM._denoise_to_x0(rgb_t, self.rgb_prompt_latent)
            rgb_denoise = rgb_denoise.permute(0, 2, 3, 1)
        self._x0_rectification(rgb_denoise, self.n_gsopt_iters)
        return rgb_t, rgb_noise_pr

    def _step_diffusion_rectification(
        self, rgb_t: torch.Tensor, rgb_noise_pr: torch.Tensor, temp_rgb_fn: str | None
    ) -> None:
        """Re-render refined scene and use it to rectify the diffusion noise, then step forward."""
        with torch.no_grad():
            x0_rect = []
            for frame in self.refine_frames:
                re_render_rgb, _, _ = self.refine_GS._render(frame)
                x0_rect.append(re_render_rgb.permute(2, 0, 1)[None])
            x0_rect = torch.cat(x0_rect, dim=0)
        if temp_rgb_fn is not None:
            random_idx = np.random.randint(0, max(1, len(self.refine_frames) // 2))
            random_frame = self.refine_frames[random_idx]
            rgb, _, _ = self.refine_GS._render(random_frame)
            rgb_np = (rgb.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            PIL.Image.fromarray(rgb_np).save(temp_rgb_fn)
        self.RGB_LCM._step_denoise(rgb_t, rgb_noise_pr, x0_rect, rect_w=self.rect_w)

    def __call__(self, temp_rgb_fn: str | None = None) -> Gaussian_Scene:
        self._pre_process()
        self._mv_init()
        for step in tqdm.tqdm(range(self.steps), desc="Fine refinement"):
            rgb_t, rgb_noise_pr = self._step_gaussian_optimization(step)
            self._step_diffusion_rectification(rgb_t, rgb_noise_pr, temp_rgb_fn)
        scene: Gaussian_Scene = self.refine_GS.GS
        for gf in scene.gaussian_frames:
            gf._require_grad(False)
        return scene
