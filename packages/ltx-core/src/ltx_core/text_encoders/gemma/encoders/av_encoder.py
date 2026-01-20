from typing import NamedTuple

import torch
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.text_encoders.gemma.embeddings_connector import (
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoderModelBase
from ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer


class AVGemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor
    attention_mask: torch.Tensor


class AVGemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    AVGemma Text Encoder Model.
    This class combines the tokenizer, Gemma model, feature extractor from base class and a
    video and audio embeddings connectors to provide a preprocessing for audio-visual pipeline.
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            feature_extractor_linear=feature_extractor_linear,
            tokenizer=tokenizer,
            model=model,
            dtype=dtype,
        )
        self.embeddings_connector = embeddings_connector.to(dtype=dtype)
        self.audio_embeddings_connector = audio_embeddings_connector.to(dtype=dtype)

    def _run_connectors(
        self, encoded_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        encoded, encoded_connector_attention_mask = self.embeddings_connector(
            encoded_input,
            connector_attention_mask,
        )

        # restore the mask values to int64
        attention_mask = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
        attention_mask = attention_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * attention_mask

        encoded_for_audio, _ = self.audio_embeddings_connector(encoded_input, connector_attention_mask)

        return encoded, encoded_for_audio, attention_mask.squeeze(-1)

    def forward(self, text: str, padding_side: str = "left") -> AVGemmaEncoderOutput:
        encoded_inputs, attention_mask = self._preprocess_text(text, padding_side)
        video_encoding, audio_encoding, attention_mask = self._run_connectors(encoded_inputs, attention_mask)
        return AVGemmaEncoderOutput(video_encoding, audio_encoding, attention_mask)


class AVGemmaTextEncoderModelConfigurator(ModelConfigurator[AVGemmaTextEncoderModel]):
    @classmethod
    def from_config(cls: type["AVGemmaTextEncoderModel"], config: dict) -> "AVGemmaTextEncoderModel":
        feature_extractor_linear = GemmaFeaturesExtractorProjLinear.from_config(config)
        embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        audio_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        return AVGemmaTextEncoderModel(
            feature_extractor_linear=feature_extractor_linear,
            embeddings_connector=embeddings_connector,
            audio_embeddings_connector=audio_embeddings_connector,
        )


def _naive_weight_or_bias_downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """
    Downcast the weight or bias to the float8_e4m3fn dtype.
    """
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


AV_GEMMA_TEXT_ENCODER_KEY_OPS = (
    SDOps("AV_GEMMA_TEXT_ENCODER_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
)

AV_GEMMA_TEXT_ENCODER_FP8_KEY_OPS = (
    SDOps("AV_GEMMA_TEXT_ENCODER_FP8_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
    .with_kv_operation(key_prefix="feature_extractor_linear.", key_suffix=".weight", operation=_naive_weight_or_bias_downcast)
    .with_kv_operation(key_prefix="feature_extractor_linear.", key_suffix=".bias", operation=_naive_weight_or_bias_downcast)
    .with_kv_operation(key_prefix="embeddings_connector.", key_suffix=".proj.weight", operation=_naive_weight_or_bias_downcast)
    .with_kv_operation(key_prefix="embeddings_connector.", key_suffix=".proj.bias", operation=_naive_weight_or_bias_downcast)
    .with_kv_operation(key_prefix="audio_embeddings_connector.", key_suffix=".proj.weight", operation=_naive_weight_or_bias_downcast)
    .with_kv_operation(key_prefix="audio_embeddings_connector.", key_suffix=".proj.bias", operation=_naive_weight_or_bias_downcast)
)
