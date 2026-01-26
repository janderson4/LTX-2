import logging
import os
import time
from collections.abc import Iterator
from dataclasses import replace

import torch

from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.is_video_only = os.environ.get("VIDEO_ONLY", "false").lower() == "true"
        if self.is_video_only:
            logging.info("Running in VIDEO_ONLY mode. Skipping all audio and audio-visual cross-attention.")

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
            is_video_only=self.is_video_only,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor | None]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.model_ledger.text_encoder()
        if enhance_prompt:
            prompt = generate_enhanced_prompt(text_encoder, prompt, images[0][0] if len(images) > 0 else None)
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()

        # Prepare latent indices for conditioning. The input images use pixel indices,
        # but the latent space is temporally downsampled.
        t_scale = self.pipeline_components.video_scale_factors.time
        latent_images = []
        for path, pixel_idx, strength in images:
            latent_idx = pixel_idx // t_scale
            latent_images.append((path, latent_idx, strength))

        # Stage 1: Initial low resolution video generation.
        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState | None]:
            base_denoise_fn = simple_denoising_func(
                video_context=video_context,
                audio_context=audio_context,
                transformer=transformer,  # noqa: F821
            )

            def timed_denoise_fn(vs, as_, s, step_idx):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                res = base_denoise_fn(vs, as_, s, step_idx)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                print(f"Diffusion step {step_idx}: {end_time - start_time:.4f}s")
                return res

            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=timed_denoise_fn,
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=latent_images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        torch.cuda.synchronize()
        print(f"Stage 1 VAE encoding: {time.perf_counter() - start_time:.4f}s")

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.model_ledger.spatial_upsampler(),
        )
        torch.cuda.synchronize()
        print(f"VAE upscaling: {time.perf_counter() - start_time:.4f}s")

        stage_2_output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate
        )
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=latent_images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        if os.environ.get("STAGE2_USE_KEYFRAME_CONDITIONING", "false").lower() == "true":
            keyframe_conditionings = image_conditionings_by_adding_guiding_latent(
                images=latent_images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )
            stage_2_conditionings.extend(keyframe_conditionings)

        torch.cuda.synchronize()
        print(f"Stage 2 VAE encoding: {time.perf_counter() - start_time:.4f}s")

        # Replace conditioned frames in upscaled_video_latent with full resolution encoded latents.
        # This ensures that high-frequency details from the original images are preserved.
        for cond in stage_2_conditionings:
            if isinstance(cond, VideoConditionByLatentIndex):
                # cond.latent_idx is already a latent index because we used latent_images
                upscaled_video_latent[
                    :, :, cond.latent_idx : cond.latent_idx + 1
                ] = cond.latent

        torch.cuda.synchronize()
        cleanup_memory()

        if os.environ.get("LTX_SKIP_STAGE2_DENOISING", "false").lower() == "true":
            # Skip the second denoising pass and use the upscaled latent directly.
            video_state = replace(video_state, latent=upscaled_video_latent)
        else:
            num_stage2_steps = int(os.environ.get("NUM_STAGE2_STEPS", "3"))
            stage_2_sigmas = DISTILLED_SIGMA_VALUES[-(num_stage2_steps + 1) :]
            stage_2_sigmas = torch.Tensor(stage_2_sigmas).to(self.device)
            noise_rescale = float(os.environ.get("STAGE2_ADD_NOISE_RESCALE", "1.0"))
            video_state, audio_state = denoise_audio_video(
                output_shape=stage_2_output_shape,
                conditionings=stage_2_conditionings,
                noiser=noiser,
                sigmas=stage_2_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop,
                components=self.pipeline_components,
                dtype=dtype,
                device=self.device,
                noise_scale=stage_2_sigmas[0].item() * noise_rescale,
                initial_video_latent=upscaled_video_latent,
                initial_audio_latent=audio_state.latent if audio_state is not None else None,
            )

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        vae_decoder = self.model_ledger.video_decoder()

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        decoded_video = vae_decode_video(
            video_state.latent, vae_decoder, tiling_config, generator
        )
        # Note: vae_decode_video returns an Iterator if tiling is enabled, or a single tensor if not.
        # We need to consume the iterator to measure full decoding time if tiling is used.
        if tiling_config is not None:
            # If it's an iterator, we'll need to wrap it to time individual chunk decodes if desired,
            # or just time the whole process by converting to list or similar.
            # But the return type of __call__ is (Iterator[torch.Tensor], torch.Tensor).
            # To measure full time accurately while maintaining the generator pattern, 
            # we might just time the first chunk or provide a wrapper.
            # Let's wrap the decoded_video iterator if it is one.
            def timed_iterator(it):
                for i, val in enumerate(it):
                    torch.cuda.synchronize()
                    chunk_start = time.perf_counter()
                    yield val
                    torch.cuda.synchronize()
                    print(f"VAE video decoding chunk {i}: {time.perf_counter() - chunk_start:.4f}s")
            
            decoded_video = timed_iterator(decoded_video)
        else:
            torch.cuda.synchronize()
            print(f"VAE video decoding: {time.perf_counter() - start_time:.4f}s")

        decoded_audio = None
        if not self.is_video_only and audio_state is not None:
            torch.cuda.synchronize()
            audio_start_time = time.perf_counter()
            decoded_audio = vae_decode_audio(
                audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
            )
            torch.cuda.synchronize()
            print(f"VAE audio decoding: {time.perf_counter() - audio_start_time:.4f}s")
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
