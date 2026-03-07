from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from jaxtyping import BFloat16, UInt8
from PIL import Image

from vistadream.flux.model import Flux
from vistadream.flux.modules.autoencoder import AutoEncoder
from vistadream.flux.sampling import denoise, get_noise, get_schedule, prepare_fill_empty_prompt, unpack
from vistadream.flux.util import load_ae, load_flow_model


class _DiffusersFluxWrapper:
    """Wraps diffusers FluxTransformer2DModel to match the custom Flux forward interface."""

    # FLUX: 16 latent channels packed with patch_size=2 → 64 packed features per token.
    # The Fill model outputs all in_channels (384) but only the first 64 are the velocity
    # prediction for the noisy latent; the rest correspond to fixed conditioning channels.
    def __init__(self, transformer) -> None:
        self.transformer = transformer

    def __call__(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None):
        result = self.transformer(
            hidden_states=img,
            encoder_hidden_states=txt,
            pooled_projections=y,
            timestep=timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )
        return result[0]

    def to(self, device):
        self.transformer = self.transformer.to(device)
        return self

    def cpu(self):
        return self.to("cpu")


@dataclass
class FluxInpaintingConfig:
    offload: bool = True
    num_steps: int = 25
    guidance: int | float = 30.0
    seed: int = 42
    model_name: str = "flux-dev-fill"
    ckpt_path: str | None = None  # override ckpt path in util.configs if set
    # quantized mode: use diffusers + bitsandbytes NF4 (~6-8GB VRAM)
    use_quantized: bool = False
    hf_model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    # GGUF mode: local GGUF quantized file, bypasses HuggingFace auth entirely
    gguf_path: str | None = None


class FluxInpainting:
    def __init__(self, config: FluxInpaintingConfig) -> None:
        self.config: FluxInpaintingConfig = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)
        self._load_model()

    def _load_model(self):
        if self.config.gguf_path is not None:
            self._load_gguf_model()
        elif self.config.use_quantized:
            self._load_quantized_model()
        else:
            self._load_original_model()

    def _load_original_model(self):
        from vistadream.flux.util import configs

        if self.config.ckpt_path is not None:
            configs[self.config.model_name].ckpt_path = self.config.ckpt_path
        self.model: Flux = load_flow_model(self.config.model_name, device="cpu" if self.config.offload else self.torch_device)
        self.ae: AutoEncoder = load_ae(self.config.model_name, device="cpu" if self.config.offload else self.torch_device)
        self._pipe = None

    def _load_gguf_model(self):
        import json
        from pathlib import Path

        from diffusers import FluxTransformer2DModel
        from diffusers.quantizers.quantization_config import GGUFQuantizationConfig

        # FluxTransformer2DModel config for the Fill model (in_channels=384 = img + mask concat).
        # Provided locally to avoid any HuggingFace network requests.
        _FILL_TRANSFORMER_CONFIG = {
            "_class_name": "FluxTransformer2DModel",
            "attention_head_dim": 128,
            "axes_dims_rope": [16, 56, 56],
            "guidance_embeds": True,
            "in_channels": 384,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 768,
        }
        config_dir = Path(self.config.gguf_path).parent / "transformer_config"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(_FILL_TRANSFORMER_CONFIG, f, indent=2)

        print(f"[INFO] Loading GGUF Flux Fill model from {self.config.gguf_path} ...")
        transformer = FluxTransformer2DModel.from_single_file(
            self.config.gguf_path,
            config=str(config_dir),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        self.model: _DiffusersFluxWrapper = _DiffusersFluxWrapper(transformer)
        self.ae: AutoEncoder = load_ae(self.config.model_name, device="cpu" if self.config.offload else self.torch_device)
        self._pipe = None
        print("[INFO] GGUF model loaded.")

    def _load_quantized_model(self):
        from diffusers import FluxFillPipeline, PipelineQuantizationConfig

        print(f"[INFO] Loading quantized Flux Fill model (NF4) from {self.config.hf_model_id} ...")
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer"],
        )
        self._pipe = FluxFillPipeline.from_pretrained(
            self.config.hf_model_id,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        ).to(self.torch_device)
        print("[INFO] Quantized model loaded.")

    def __call__(
        self,
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
        mask: UInt8[np.ndarray, "h w"],
    ) -> Image.Image:
        if self.config.use_quantized and self._pipe is not None:
            return self._call_quantized(rgb_hw3, mask)
        return self._call_original(rgb_hw3, mask)

    def _call_quantized(
        self,
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
        mask: UInt8[np.ndarray, "h w"],
    ) -> Image.Image:
        input_image: Image.Image = Image.fromarray(rgb_hw3)
        mask_image: Image.Image = Image.fromarray(mask)  # white=255 means inpaint
        result: Image.Image = self._pipe(
            prompt="",
            image=input_image,
            mask_image=mask_image,
            height=rgb_hw3.shape[0],
            width=rgb_hw3.shape[1],
            num_inference_steps=self.config.num_steps,
            guidance_scale=self.config.guidance,
            generator=torch.Generator(device=self.torch_device).manual_seed(self.config.seed),
        ).images[0]
        return result

    @torch.inference_mode
    def _call_original(
        self,
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
        mask: UInt8[np.ndarray, "h w"],
    ) -> Image.Image:
        height: int = rgb_hw3.shape[0]
        width: int = rgb_hw3.shape[1]
        x: BFloat16[torch.Tensor, "batch channels latent_height latent_width"] = get_noise(
            num_samples=1,
            height=height,
            width=width,
            device=self.torch_device,
            dtype=torch.bfloat16,
            seed=self.config.seed,
        )

        if self.config.offload:
            self.ae = self.ae.to(self.torch_device)

        inp: dict[str, torch.Tensor] = prepare_fill_empty_prompt(
            x,
            prompt="",
            ae=self.ae,
            img_cond=rgb_hw3,
            mask=mask,
        )

        timesteps: list[float] = get_schedule(self.config.num_steps, inp["img"].shape[1], shift=True)

        if self.config.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.torch_device)

        x = denoise(self.model, **inp, timesteps=timesteps, guidance=self.config.guidance)

        if self.config.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        torch.cuda.empty_cache()
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        inpainted_image: Image.Image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return inpainted_image
