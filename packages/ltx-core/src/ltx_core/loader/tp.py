import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from ltx_core.model.transformer.model import LTXModel


def maybe_init_dist():
    if not dist.is_initialized():
        # Only initialize if torchrun or similar set up the environment
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group("nccl")
            # Use LOCAL_RANK to ensure we set the correct device on the node
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            return True
    return False


_MESH_CACHE = {}


def get_tp_mesh(tp_size):
    global _MESH_CACHE
    if tp_size not in _MESH_CACHE:
        _MESH_CACHE[tp_size] = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
    return _MESH_CACHE[tp_size]


def parallelize_ltx_model(model: LTXModel, device_mesh):
    """
    Parallelize LTXModel using Tensor Parallelism (Megatron-style).
    This implementation keeps the hidden state REPLICATED at the boundaries of
    sub-blocks (Attention/FFN) to ensure compatibility with AdaLN modulation
    and skip connections without additional communication overhead.
    """
    tp_mesh = device_mesh["tp"]

    # Parallelize transformer blocks
    for block in model.transformer_blocks:
        # 1. Parallelize Video Attention
        if hasattr(block, "attn1"):
            parallelize_module(
                block.attn1,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(),
                },
            )
        if hasattr(block, "attn2"):
            parallelize_module(
                block.attn2,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(),
                },
            )

        # 2. Parallelize Audio Attention
        if hasattr(block, "audio_attn1"):
            parallelize_module(
                block.audio_attn1,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(),
                },
            )
        if hasattr(block, "audio_attn2"):
            parallelize_module(
                block.audio_attn2,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(),
                },
            )

        # 3. Parallelize Cross-Modality Attention
        if hasattr(block, "audio_to_video_attn"):
            parallelize_module(
                block.audio_to_video_attn,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(),
                },
            )
        if hasattr(block, "video_to_audio_attn"):
            parallelize_module(
                block.video_to_audio_attn,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(),
                },
            )

        # 4. Parallelize FeedForward Networks
        if hasattr(block, "ff"):
            parallelize_module(
                block.ff,
                tp_mesh,
                {
                    "net.0.proj": ColwiseParallel(),
                    "net.2": RowwiseParallel(),
                },
            )
        if hasattr(block, "audio_ff"):
            parallelize_module(
                block.audio_ff,
                tp_mesh,
                {
                    "net.0.proj": ColwiseParallel(),
                    "net.2": RowwiseParallel(),
                },
            )

    return model


def parallelize_gemma_model(model, device_mesh):
    """
    Parallelize Gemma model using Tensor Parallelism.
    This assumes a standard HF Gemma model structure.
    """
    tp_mesh = device_mesh["tp"]
    
    # Iterate through decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        # Fallback or skip if structure unknown
        return model

    for layer in layers:
        parallelize_module(
            layer,
            tp_mesh,
            {
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            },
        )
    return model


def apply_tp(model, tp_size):
    if tp_size <= 1:
        return model
    
    maybe_init_dist()
    mesh = get_tp_mesh(tp_size)
    
    if isinstance(model, LTXModel):
        return parallelize_ltx_model(model, mesh)
    elif hasattr(model, "velocity_model") and isinstance(model.velocity_model, LTXModel):
        model.velocity_model = parallelize_ltx_model(model.velocity_model, mesh)
        return model
    elif hasattr(model, "model") and "Gemma" in model.__class__.__name__:
        return parallelize_gemma_model(model, mesh)
    elif "Gemma" in model.__class__.__name__:
        return parallelize_gemma_model(model, mesh)
    
    return model
