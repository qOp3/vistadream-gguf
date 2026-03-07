import argparse
import datetime
import os
import random
import time

import gradio as gr
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, UInt8
from PIL import Image

from vistadream.flux.sampling import denoise, get_noise, get_schedule, prepare_fill_empty_prompt, unpack
from vistadream.flux.util import load_ae, load_flow_model


def add_border_and_mask(image, zoom_all=1.0, zoom_left=0, zoom_right=0, zoom_up=0, zoom_down=0, overlap=0):
    """Adds a black border around the image with individual side control and mask overlap"""
    orig_width, orig_height = image.size

    # Calculate padding for each side (in pixels)
    left_pad = int(orig_width * zoom_left)
    right_pad = int(orig_width * zoom_right)
    top_pad = int(orig_height * zoom_up)
    bottom_pad = int(orig_height * zoom_down)

    # Calculate overlap in pixels
    overlap_left = int(orig_width * overlap)
    overlap_right = int(orig_width * overlap)
    overlap_top = int(orig_height * overlap)
    overlap_bottom = int(orig_height * overlap)

    # If using the all-sides zoom, add it to each side
    if zoom_all > 1.0:
        extra_each_side = (zoom_all - 1.0) / 2
        left_pad += int(orig_width * extra_each_side)
        right_pad += int(orig_width * extra_each_side)
        top_pad += int(orig_height * extra_each_side)
        bottom_pad += int(orig_height * extra_each_side)

    # Calculate new dimensions (ensure they're multiples of 32)
    new_width = 32 * round((orig_width + left_pad + right_pad) / 32)
    new_height = 32 * round((orig_height + top_pad + bottom_pad) / 32)

    # Create new image with black border
    bordered_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    # Paste original image in position
    paste_x = left_pad
    paste_y = top_pad
    bordered_image.paste(image, (paste_x, paste_y))

    # Create mask (white where the border is, black where the original image was)
    mask = Image.new("L", (new_width, new_height), 255)  # White background
    # Paste black rectangle with overlap adjustment
    mask.paste(
        0,
        (
            paste_x + overlap_left,  # Left edge moves right
            paste_y + overlap_top,  # Top edge moves down
            paste_x + orig_width - overlap_right,  # Right edge moves left
            paste_y + orig_height - overlap_bottom,  # Bottom edge moves up
        ),
    )

    return bordered_image, mask


def get_models(name: str, device: torch.device, offload: bool):
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    # nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    return model, ae  # , nsfw_classifier


def resize(img: Image.Image, min_mp: float = 0.5, max_mp: float = 2.0) -> Image.Image:
    """
    Resize an image to ensure its megapixel count falls within a specified range and its dimensions are multiples of 32.

    If the image's megapixel count is already within the specified range (`min_mp` to `max_mp`), the function will only adjust the dimensions to the nearest multiples of 32 if necessary. Otherwise, it will scale the image so that its megapixel count is within the range, and then adjust the dimensions to the nearest multiples of 32.

    Args:
        img (Image.Image): The input PIL Image to be resized.
        min_mp (float, optional): Minimum allowed megapixels for the output image. Defaults to 0.5.
        max_mp (float, optional): Maximum allowed megapixels for the output image. Defaults to 2.0.

    Returns:
        Image.Image: The resized PIL Image with dimensions as multiples of 32 and megapixels within the specified range.
    """
    width, height = img.size
    mp = (width * height) / 1_000_000  # Current megapixels

    if min_mp <= mp <= max_mp:
        # Even if MP is in range, ensure dimensions are multiples of 32
        new_width = int(32 * round(width / 32))
        new_height = int(32 * round(height / 32))
        if new_width != width or new_height != height:
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img

    # Calculate scaling factor
    scale = (min_mp / mp) ** 0.5 if mp < min_mp else (max_mp / mp) ** 0.5

    new_width = int(32 * round(width * scale / 32))
    new_height = int(32 * round(height * scale / 32))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


@torch.no_grad
def get_flux_fill_res(
    tmp_img, tmp_mask, prompt, height, width, num_steps, guidance, model, ae, torch_device, seed, offload
):
    x = get_noise(
        1,
        height,
        width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    if offload:
        ae = ae.to(torch_device)

    # Load images from paths and convert to numpy arrays
    img_cond_pil: Image.Image = Image.open(tmp_img).convert("RGB")
    img_cond: UInt8[np.ndarray, "h w 3"] = np.array(img_cond_pil)

    mask_pil: Image.Image = Image.open(tmp_mask).convert("L")
    mask: UInt8[np.ndarray, "h w"] = np.array(mask_pil)

    inp = prepare_fill_empty_prompt(
        x,
        prompt=prompt,
        ae=ae,
        img_cond=img_cond,
        mask=mask,
    )

    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=True)

    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    return x


def build_ui(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = True,
    use_quantized: bool = False,
    hf_model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
):
    torch_device = torch.device(device)

    if use_quantized:
        from vistadream.ops.flux import FluxInpainting, FluxInpaintingConfig
        flux_inpainter = FluxInpainting(FluxInpaintingConfig(
            use_quantized=True,
            hf_model_id=hf_model_id,
        ))
        model, ae = None, None
    else:
        name = "flux-dev-fill"
        model, ae = get_models(name, device=torch_device, offload=offload)
        flux_inpainter = None

    def get_res(image, expansion_percent) -> Image.Image:
        max_dimension = 1024
        width, height = image.size
        print(f"Original image size: {width}x{height}")

        scale = min(max_dimension / width, max_dimension / height)
        if scale < 1.0:
            new_width = int(32 * round(width * scale / 32))
            new_height = int(32 * round(height * scale / 32))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            image = resize(image)

        width, height = image.size
        print(f"Resized image size: {width}x{height}")

        border_percent = expansion_percent / 200.0
        print(f"Border expansion: {expansion_percent}% total ({border_percent * 100:.1f}% per side)")
        image, mask = add_border_and_mask(
            image,
            zoom_all=1.0,
            zoom_left=border_percent,
            zoom_right=border_percent,
            zoom_up=border_percent,
            zoom_down=border_percent,
            overlap=0,
        )

        t0: float = time.perf_counter()

        if use_quantized:
            img: Image.Image = flux_inpainter(
                rgb_hw3=np.array(image.convert("RGB")),
                mask=np.array(mask.convert("L")),
            )
        else:
            output_dir = "./tmp"
            os.makedirs(output_dir, exist_ok=True)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tag = f"{current_time}_{random.randint(10000, 99999)}"
            tmp_img = os.path.join(output_dir, f"{tag}_image.png")
            tmp_mask = os.path.join(output_dir, f"{tag}_mask.png")
            image.save(tmp_img)
            mask.save(tmp_mask)

            x = get_flux_fill_res(
                tmp_img, tmp_mask, "", height, width, 25, 30.0, model, ae, torch_device, 42, offload
            )
            torch.cuda.empty_cache()
            x_clamped: torch.Tensor = x.clamp(-1, 1)
            x_rearranged: Float[torch.Tensor, "h w c"] = rearrange(x_clamped[0], "c h w -> h w c")
            img_array: UInt8[np.ndarray, "h w c"] = (127.5 * (x_rearranged + 1.0)).cpu().byte().numpy()
            img = Image.fromarray(img_array)

        t1: float = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s")

        return img

    with gr.Blocks() as demo, gr.Column():
        # Simple image upload for outpainting
        input_image = gr.Image(
            type="pil",
            sources=["upload", "clipboard"],
            label="Upload Image for Outpainting",
        )

        # Slider for border expansion percentage
        expansion_slider = gr.Slider(
            minimum=5,
            maximum=50,
            value=20,
            step=1,
            label="Border Expansion (%)",
            info="Total percentage to expand the image (distributed evenly on all sides)",
        )

        run_btn = gr.Button("Run Outpainting")

        with gr.Row():
            flux_res = gr.Image(label="Outpainted Result")

        run_btn.click(fn=get_res, inputs=[input_image, expansion_slider], outputs=[flux_res])

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-quantized", action="store_true", help="Use NF4 quantized model via diffusers (~6-8GB VRAM)")
    parser.add_argument("--hf-model-id", type=str, default="black-forest-labs/FLUX.1-Fill-dev", help="HuggingFace model ID for quantized loading")
    parser.add_argument("--no-offload", action="store_true", help="Disable CPU offload (requires more VRAM)")
    args = parser.parse_args()

    build_ui(offload=not args.no_offload, use_quantized=args.use_quantized, hf_model_id=args.hf_model_id)
