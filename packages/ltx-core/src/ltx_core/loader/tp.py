import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.model import LTXModel


def maybe_init_dist():
    if not dist.is_initialized():
        # Only initialize if torchrun or similar set up the environment
        if "RANK" in torch.os.environ and "WORLD_SIZE" in torch.os.environ:
            dist.init_process_group("nccl")
            rank = dist.get_rank()
            torch.cuda.set_device(rank)
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
    Parallelize LTXModel using Tensor Parallelism.
    """
    tp_mesh = device_mesh["tp"]

    # Parallelize transformer blocks
    for block in model.transformer_blocks:
        # Parallelize video attention
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
        # Parallelize audio attention
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
        # Parallelize cross attention
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
        
        # Parallelize FeedForward
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

    # Parallelize input/output projections if they are large enough
    if hasattr(model, "patchify_proj"):
        parallelize_module(model, tp_mesh, {"patchify_proj": ColwiseParallel()})
    if hasattr(model, "proj_out"):
        parallelize_module(model, tp_mesh, {"proj_out": RowwiseParallel()})
    
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

