import torch
from diffusers import LCMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *


class Hack_SDPipe_Stepwise(StableDiffusionPipeline):

    @torch.no_grad()
    def _use_lcm(self, use=True, ckpt="latent-consistency/lcm-lora-sdv1-5"):
        if use:
            self.use_lcm = True
            adapter_id = ckpt
            self.scheduler = LCMScheduler.from_config(self.scheduler.config)
            self._guidance_scale = 0.0
            self.load_lora_weights(adapter_id)
            self.fuse_lora()
        else:
            self.use_lcm = False
            self._guidance_scale = 7.5

    @torch.no_grad()
    def re_init(self, num_inference_steps=50):
        eta = 0.0
        timesteps = None
        generator = None
        self._clip_skip = None
        self._interrupt = False
        self._guidance_rescale = 0.0
        self.added_cond_kwargs = None
        self._cross_attention_kwargs = None
        self._do_classifier_free_guidance = self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

        batch_size = 1
        device = self._execution_device

        self.timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self.extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * 1)
            self.timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device)

    @torch.no_grad()
    def _encode_text_prompt(self, prompt, negative_prompt="fake,ugly,unreal"):
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self._execution_device,
            1,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def _step_noise(self, latents, time_step, prompt_embeds):
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, time_step)
        noise_pred = self.unet(
            latent_model_input,
            time_step,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=self.timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=self.added_cond_kwargs,
            return_dict=False,
        )[0]
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
        return noise_pred

    def _encode(self, input):
        """input: B3HW, return: B4H'W'"""
        h = self.vae.encoder(input)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        latent = mean * self.vae.config.scaling_factor
        return latent

    def _decode(self, latent):
        """input: B4H'W', return: B3HW"""
        latent = latent / self.vae.config.scaling_factor
        z = self.vae.post_quant_conv(latent)
        output = self.vae.decoder(z)
        return output

    def _solve_x0_full_step(self, latents, noise_pred, t):
        self.alpha_t = torch.sqrt(self.scheduler.alphas_cumprod).to(t.device)
        self.sigma_t = torch.sqrt(1 - self.scheduler.alphas_cumprod).to(t.device)
        a_t, s_t = self.alpha_t[t], self.sigma_t[t]
        x0_latents = (latents - s_t * noise_pred) / a_t
        x0 = self._decode(x0_latents)
        return x0_latents, x0

    def _solve_x0(self, latents, noise_pred, t):
        x0_latents = self.scheduler.step(noise_pred, t.squeeze(), latents)
        self.scheduler._step_index -= 1
        x0_latents = x0_latents.denoised if self.use_lcm else x0_latents.pred_original_sample
        x0 = self._decode(x0_latents)
        return x0_latents, x0

    def _step_denoise(self, latents, noise_pred, t):
        latents = self.scheduler.step(noise_pred, t.squeeze(), latents).prev_sample
        return latents

    def xt_x0_noise(
        self,
        xt_latents: torch.Tensor,
        x0_latents: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=xt_latents.dtype, device=xt_latents.device)
        timesteps = timesteps.to(xt_latents.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(xt_latents.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(xt_latents.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = (xt_latents - sqrt_alpha_prod * x0_latents) / sqrt_one_minus_alpha_prod
        return noise

    def _solve_noise_given_x0_latent(self, latents, x0_latents, t):
        noise = self.xt_x0_noise(latents, x0_latents, t)
        if self.scheduler.config.prediction_type == "epsilon":
            noise = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(x0_latents, noise, t)
        return noise
