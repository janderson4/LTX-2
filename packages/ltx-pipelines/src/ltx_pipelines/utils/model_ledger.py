from dataclasses import replace

import torch

from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.tp import apply_tp, maybe_init_dist
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
    UPCAST_DURING_INFERENCE,
    LTXModelConfigurator,
    X0Model,
    amend_forward_with_fp8_compute,
    amend_forward_with_upcast,
    is_transformer_engine_available,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.text_encoders.gemma import (
    AV_GEMMA_TEXT_ENCODER_FP8_KEY_OPS,
    AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    AVGemmaTextEncoderModel,
    AVGemmaTextEncoderModelConfigurator,
    module_ops_from_gemma_root,
)


class ModelLedger:
    """
    Central coordinator for loading and building models used in an LTX pipeline.
    The ledger wires together multiple model builders (transformer, video VAE encoder/decoder,
    audio VAE decoder, vocoder, text encoder, and optional latent upsampler) and exposes
    factory methods for constructing model instances.
    ### Model Building
    Each model method (e.g. :meth:`transformer`, :meth:`video_decoder`, :meth:`text_encoder`)
    constructs a new model instance on each call. The builder uses the
    :class:`~ltx_core.loader.registry.Registry` to load weights from the checkpoint,
    instantiates the model with the configured ``dtype``, and moves it to ``self.device``.
    .. note::
        Models are **not cached**. Each call to a model method creates a new instance.
        Callers are responsible for storing references to models they wish to reuse
        and for freeing GPU memory (e.g. by deleting references and calling
        ``torch.cuda.empty_cache()``).
    ### Constructor parameters
    dtype:
        Torch dtype used when constructing all models (e.g. ``torch.bfloat16``).
    device:
        Target device to which models are moved after construction (e.g. ``torch.device("cuda")``).
    checkpoint_path:
        Path to a checkpoint directory or file containing the core model weights
        (transformer, video VAE, audio VAE, text encoder, vocoder). If ``None``, the
        corresponding builders are not created and calling those methods will raise
        a :class:`ValueError`.
    gemma_root_path:
        Base path to Gemma-compatible CLIP/text encoder weights. Required to
        initialize the text encoder builder; if omitted, :meth:`text_encoder` cannot be used.
    spatial_upsampler_path:
        Optional path to a latent upsampler checkpoint. If provided, the
        :meth:`spatial_upsampler` method becomes available; otherwise calling it raises
        a :class:`ValueError`.
    loras:
        Optional collection of LoRA configurations (paths, strengths, and key operations)
        that are applied on top of the base transformer weights when building the model.
    registry:
        Optional :class:`Registry` instance for weight caching across builders.
        Defaults to :class:`DummyRegistry` which performs no cross-builder caching.
    use_fp8:
        If ``True``, enables FP8 mode for transformer and text encoder, using true FP8 compute
        when supported and falling back to memory-only FP8 otherwise.
    ### Creating Variants
    Use :meth:`with_loras` to create a new ``ModelLedger`` instance that includes
    additional LoRA configurations while sharing the same registry for weight caching.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: LoraPathStrengthAndSDOps | None = None,
        registry: Registry | None = None,
        use_fp8: bool = False,
        compile: bool = False,
        tp_degree: int = 1,
    ):
        if tp_degree > 1:
            maybe_init_dist()
            
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras or ()
        self.registry = registry or DummyRegistry()
        self.use_fp8 = use_fp8
        self.compile = compile
        self.tp_degree = tp_degree
        self._fp8_available = self._check_fp8_compute_support()
        # True FP8 compute is only enabled when TE is available and TP is not used.
        self.use_fp8_compute = self.use_fp8 and self._fp8_available and self.tp_degree == 1
        self.build_model_builders()

    def build_model_builders(self) -> None:
        if self.checkpoint_path is not None:
            self.transformer_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
            )

            self.vae_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vae_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.audio_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            self.vocoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )

            if self.gemma_root_path is not None:
                # When using FP8 compute, load weights in bfloat16 - let TE handle quantization
                use_naive_fp8 = self.use_fp8 and not self.use_fp8_compute
                self.text_encoder_builder = Builder(
                    model_path=self.checkpoint_path,
                    model_class_configurator=AVGemmaTextEncoderModelConfigurator,
                    model_sd_ops=AV_GEMMA_TEXT_ENCODER_FP8_KEY_OPS if use_naive_fp8 else AV_GEMMA_TEXT_ENCODER_KEY_OPS,
                    registry=self.registry,
                    module_ops=module_ops_from_gemma_root(self.gemma_root_path, quantize_fp8=use_naive_fp8),
                )

        if self.spatial_upsampler_path is not None:
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry) or self.registry is None:
            return self.device
        else:
            return torch.device("cpu")

    def _check_fp8_compute_support(self) -> bool:
        """Check if GPU supports true FP8 compute (H100/H200/Blackwell)"""
        if not is_transformer_engine_available():
            return False
        if not torch.cuda.is_available():
            return False
        try:
            capability = torch.cuda.get_device_capability(self.device)
            return capability[0] >= 9  # Hopper (9.0) or Blackwell (10.0)
        except Exception:
            return False

    def _apply_fp8_compute(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply true FP8 compute using Transformer-Engine FP8 kernels"""
        if not self.use_fp8_compute or not self._fp8_available:
            return model
        return amend_forward_with_fp8_compute(model)

    def with_loras(self, loras: LoraPathStrengthAndSDOps) -> "ModelLedger":
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            loras=(*self.loras, *loras),
            registry=self.registry,
            use_fp8=self.use_fp8,
            compile=self.compile,
            tp_degree=self.tp_degree,
        )

    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )
        if self.use_fp8:
            if self.use_fp8_compute:
                # True FP8 compute: 
                # 1. Load in BF16
                base_model = self.transformer_builder.build(device=self._target_device(), dtype=self.dtype)
                base_model = base_model.to(self.device).eval()
                model = X0Model(base_model)
                
                # 2. Apply FP8 Compute with Transformer-Engine kernels
                model.velocity_model = self._apply_fp8_compute(model.velocity_model)

                # 3. Apply TP after FP8 module conversion
                if self.tp_degree > 1:
                    model = apply_tp(model, self.tp_degree)
            else:
                # Memory-only FP8: use upcast approach
                fp8_builder = replace(
                    self.transformer_builder,
                    module_ops=(UPCAST_DURING_INFERENCE,),
                    model_sd_ops=LTXV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP,
                )
                model = X0Model(fp8_builder.build(device=self._target_device())).to(self.device).eval()
                if self.tp_degree > 1:
                    model = apply_tp(model, self.tp_degree)
        else:
            model = (
                X0Model(self.transformer_builder.build(device=self._target_device(), dtype=self.dtype))
                .to(self.device)
                .eval()
            )
            if self.tp_degree > 1:
                model = apply_tp(model, self.tp_degree)

        if self.compile:
            model = torch.compile(model)
        return model

    def video_decoder(self) -> VideoDecoder:
        if not hasattr(self, "vae_decoder_builder"):
            raise ValueError(
                "Video decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        model = self.vae_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        if self.compile:
            model = torch.compile(model)
        return model

    def video_encoder(self) -> VideoEncoder:
        if not hasattr(self, "vae_encoder_builder"):
            raise ValueError(
                "Video encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        model = self.vae_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        if self.compile:
            model = torch.compile(model)
        return model

    def text_encoder(self) -> AVGemmaTextEncoderModel:
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError(
                "Text encoder not initialized. Please provide a checkpoint path and gemma root path to the "
                "ModelLedger constructor."
            )

        model = self.text_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()

        # 1. Apply FP8 Compute or Upcast
        if self.use_fp8:
            if self.use_fp8_compute:
                # True FP8 compute
                model = self._apply_fp8_compute(model)
            else:
                # Memory-only FP8
                model = amend_forward_with_upcast(model)
        # 2. Apply TP after FP8 module conversion
        if self.tp_degree > 1:
            model = apply_tp(model, self.tp_degree)

        # Text encoder is often left uncompiled due to string processing/tokenization, 
        # but the heavy feature extraction can be compiled.
        if self.compile:
            model = torch.compile(model)
        return model

    def audio_decoder(self) -> AudioDecoder:
        if not hasattr(self, "audio_decoder_builder"):
            raise ValueError(
                "Audio decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        model = self.audio_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        if self.compile:
            model = torch.compile(model)
        return model

    def vocoder(self) -> Vocoder:
        if not hasattr(self, "vocoder_builder"):
            raise ValueError(
                "Vocoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        model = self.vocoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        if self.compile:
            model = torch.compile(model)
        return model

    def spatial_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "upsampler_builder"):
            raise ValueError("Upsampler not initialized. Please provide upsampler path to the ModelLedger constructor.")

        model = self.upsampler_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        if self.compile:
            model = torch.compile(model)
        return model
