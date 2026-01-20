import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
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


def shard_norm(norm, mesh):
    """
    Shard an RMSNorm or LayerNorm module's weight and bias.
    Also updates normalized_shape to match the sharded input layout.
    """
    if getattr(norm, "_tp_sharded", False):
        return

    if hasattr(norm, "weight") and norm.weight is not None:
        norm.weight.data = distribute_tensor(norm.weight.data, mesh, [Shard(-1)])
    if hasattr(norm, "bias") and norm.bias is not None:
        norm.bias.data = distribute_tensor(norm.bias.data, mesh, [Shard(-1)])

    if isinstance(norm, torch.nn.RMSNorm):
        def _get_local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to_local() if isinstance(tensor, DTensor) else tensor

        def sharded_rmsnorm(x: torch.Tensor) -> torch.Tensor:
            x_local = _get_local(x)
            global_dim = x.shape[-1] if isinstance(x, DTensor) else x_local.shape[-1] * mesh.size()
            local_sum = (x_local * x_local).sum(dim=-1, keepdim=True)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=mesh.get_group())
            inv_rms = torch.rsqrt(local_sum / global_dim + norm.eps)
            y_local = x_local * inv_rms

            if norm.weight is not None:
                y_local = y_local * _get_local(norm.weight)
            if norm.bias is not None:
                y_local = y_local + _get_local(norm.bias)

            if isinstance(x, DTensor):
                return distribute_tensor(y_local, mesh, [Shard(-1)])
            return y_local

        norm.forward = sharded_rmsnorm
    else:
        # Update the module's expected shape so the runtime check passes
        if hasattr(norm, "normalized_shape"):
            local_size = norm.weight.data.size(-1)
            norm.normalized_shape = (local_size,)

    norm._tp_sharded = True


def update_attn_heads(attn, tp_size):
    """
    Update the local head count for an Attention module after sharding.
    This ensures that attention kernels (xformers, flash_attn) see the 
    correct number of heads per GPU.
    """
    if hasattr(attn, "heads"):
        attn.heads = attn.heads // tp_size


def parallelize_ltx_model(model: LTXModel, device_mesh):
    """
    Parallelize LTXModel using Tensor Parallelism (Megatron-style).
    This implementation keeps the hidden state REPLICATED at the boundaries of
    sub-blocks (Attention/FFN) to ensure compatibility with AdaLN modulation
    and skip connections without additional communication overhead.
    """
    tp_mesh = device_mesh["tp"]
    tp_size = device_mesh.size()

    # Parallelize transformer blocks
    for block in model.transformer_blocks:
        # 1. Parallelize Video Attention
        if hasattr(block, "attn1"):
            # Shard the internal QK norms to match the col-sharded linear outputs
            shard_norm(block.attn1.q_norm, tp_mesh)
            shard_norm(block.attn1.k_norm, tp_mesh)
            update_attn_heads(block.attn1, tp_size)

            parallelize_module(
                block.attn1,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(output_layouts=Replicate()),  # Explicit All-Reduce to 4096
                },
            )
        if hasattr(block, "attn2"):
            shard_norm(block.attn2.q_norm, tp_mesh)
            shard_norm(block.attn2.k_norm, tp_mesh)
            update_attn_heads(block.attn2, tp_size)
            parallelize_module(
                block.attn2,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(output_layouts=Replicate()),
                },
            )

        # 2. Parallelize Audio Attention
        if hasattr(block, "audio_attn1"):
            shard_norm(block.audio_attn1.q_norm, tp_mesh)
            shard_norm(block.audio_attn1.k_norm, tp_mesh)
            update_attn_heads(block.audio_attn1, tp_size)
            parallelize_module(
                block.audio_attn1,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(output_layouts=Replicate()),
                },
            )
        if hasattr(block, "audio_attn2"):
            shard_norm(block.audio_attn2.q_norm, tp_mesh)
            shard_norm(block.audio_attn2.k_norm, tp_mesh)
            update_attn_heads(block.audio_attn2, tp_size)
            parallelize_module(
                block.audio_attn2,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(output_layouts=Replicate()),
                },
            )

        # 3. Parallelize Cross-Modality Attention
        if hasattr(block, "audio_to_video_attn"):
            shard_norm(block.audio_to_video_attn.q_norm, tp_mesh)
            shard_norm(block.audio_to_video_attn.k_norm, tp_mesh)
            update_attn_heads(block.audio_to_video_attn, tp_size)
            parallelize_module(
                block.audio_to_video_attn,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(output_layouts=Replicate()),
                },
            )
        if hasattr(block, "video_to_audio_attn"):
            shard_norm(block.video_to_audio_attn.q_norm, tp_mesh)
            shard_norm(block.video_to_audio_attn.k_norm, tp_mesh)
            update_attn_heads(block.video_to_audio_attn, tp_size)
            parallelize_module(
                block.video_to_audio_attn,
                tp_mesh,
                {
                    "to_q": ColwiseParallel(),
                    "to_k": ColwiseParallel(),
                    "to_v": ColwiseParallel(),
                    "to_out.0": RowwiseParallel(output_layouts=Replicate()),
                },
            )

        # 4. Parallelize FeedForward
        # We target the parent block.ff to ensure correct sharding propagation
        if hasattr(block, "ff"):
            parallelize_module(
                block.ff,
                tp_mesh,
                {
                    "net.0.proj": ColwiseParallel(),
                    "net.2": RowwiseParallel(output_layouts=Replicate()),  # Explicit All-Reduce to 4096
                },
            )
        if hasattr(block, "audio_ff"):
            parallelize_module(
                block.audio_ff,
                tp_mesh,
                {
                    "net.0.proj": ColwiseParallel(),
                    "net.2": RowwiseParallel(output_layouts=Replicate()),
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
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        layers = model.model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
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
