"""
Test: overlap FusedDispatch all-to-all communication with Linear computation.

In a typical MoE Transformer, while tokens are being dispatched to experts
(all-to-all communication), independent computation (e.g., attention linear
projections) can execute concurrently on the GPU.

DeepEP enables this by:
  1. Limiting communication to a subset of SMs (Buffer.set_num_sms)
  2. Running dispatch asynchronously (async_finish=True)
  3. Allocating outputs on the comm stream (allocate_on_comm_stream=True)

This script benchmarks two modes:
  - Sequential: dispatch waits to finish, then linear runs
  - Overlapped: dispatch runs async on comm stream, linear runs on default
    stream in parallel, then synchronizes

Run (mpirun):
    mpirun --allow-run-as-root -np 8 python3 test_overlap.py
Run (torchrun):
    torchrun --nproc_per_node=8 test_overlap.py
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from deep_ep import Buffer
from deep_ep.utils import EventHandle, EventOverlap

from fused_a2a import get_buffer, get_hidden_bytes


def _env_first(*keys, default="0"):
    """Return the first non-empty value among environment variable keys."""
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return default


def setup():
    """Initialize distributed, supporting mpirun / torchrun / srun launchers."""
    local_rank = int(_env_first(
        "LOCAL_RANK",                   # torchrun
        "OMPI_COMM_WORLD_LOCAL_RANK",   # OpenMPI
        "MV2_COMM_WORLD_LOCAL_RANK",    # MVAPICH
        "MPI_LOCALRANKID",              # MPICH / Intel MPI
        default="0",
    ))
    rank = int(_env_first(
        "RANK",                         # torchrun
        "OMPI_COMM_WORLD_RANK",         # OpenMPI
        "MV2_COMM_WORLD_RANK",          # MVAPICH
        "PMI_RANK",                     # MPICH / Intel MPI
        default="0",
    ))
    world_size = int(_env_first(
        "WORLD_SIZE",                   # torchrun
        "OMPI_COMM_WORLD_SIZE",         # OpenMPI
        "MV2_COMM_WORLD_SIZE",          # MVAPICH
        "PMI_SIZE",                     # MPICH / Intel MPI
        default="1",
    ))

    # Force-set so that PyTorch env:// rendezvous can read them
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"[setup] rank={rank}, world_size={world_size}, "
              f"local_rank={local_rank}")

    return local_rank


def run_dispatch_then_linear(buffer, x, token_indices, token_probs,
                             num_experts, linear, linear_input):
    """Sequential: synchronous dispatch followed by linear."""
    num_tokens_per_rank, num_tokens_per_rdma_rank, \
        num_tokens_per_expert, is_token_in_rank, event = \
        buffer.get_dispatch_layout(
            token_indices, num_experts,
            previous_event=None,
            async_finish=False,
        )

    recv_x, recv_token_indices, recv_token_probs, \
        num_recv_tokens_per_expert_list, handle, _ = \
        buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=False,
        )

    linear_out = linear(linear_input)
    return recv_x, linear_out, handle


def run_dispatch_overlap_linear(buffer, x, token_indices, token_probs,
                                num_experts, linear, linear_input):
    """Overlapped: async dispatch on comm stream || linear on default stream."""
    prev_event = EventOverlap(EventHandle())

    num_tokens_per_rank, num_tokens_per_rdma_rank, \
        num_tokens_per_expert, is_token_in_rank, event = \
        buffer.get_dispatch_layout(
            token_indices, num_experts,
            previous_event=prev_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )

    recv_x, recv_token_indices, recv_token_probs, \
        num_recv_tokens_per_expert_list, handle, after_event = \
        buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )

    # Linear computation overlaps with dispatch communication
    linear_out = linear(linear_input)

    # Wait for dispatch to complete before using recv_x
    after_event.current_stream_wait()

    return recv_x, linear_out, handle


def bench(fn, num_warmup=5, num_iters=20, **kwargs):
    """Benchmark a function with CUDA event timing."""
    for _ in range(num_warmup):
        fn(**kwargs)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(num_iters):
        fn(**kwargs)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_iters


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FusedDispatch + Linear overlap")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--experts-per-rank", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--num-sms", type=int, default=24,
                        help="SMs reserved for DeepEP communication kernels")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    args = parser.parse_args()

    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    num_experts = world_size * args.experts_per_rank
    ffn_hidden = args.hidden_size * 4

    # Reserve a subset of SMs for DeepEP so the rest are free for computation
    Buffer.set_num_sms(args.num_sms)

    # ---- test tensors ----
    x = torch.randn(args.num_tokens, args.hidden_size,
                     device=device, dtype=torch.bfloat16)
    token_indices = torch.randint(
        0, num_experts, (args.num_tokens, args.topk), device=device)
    token_probs = torch.randn(
        args.num_tokens, args.topk, device=device, dtype=torch.float32
    ).softmax(dim=-1)
    # Independent input for the linear layer (simulates attention-path data)
    linear_input = torch.randn(args.num_tokens, args.hidden_size,
                               device=device, dtype=torch.bfloat16)

    linear = nn.Linear(args.hidden_size, ffn_hidden, bias=False,
                       device=device, dtype=torch.bfloat16)

    group = dist.group.WORLD
    buffer = get_buffer(group, get_hidden_bytes(x))

    common_kwargs = dict(
        buffer=buffer,
        x=x,
        token_indices=token_indices,
        token_probs=token_probs,
        num_experts=num_experts,
        linear=linear,
        linear_input=linear_input,
    )

    # ---- benchmark ----
    t_seq = bench(run_dispatch_then_linear,
                  num_warmup=args.num_warmup, num_iters=args.num_iters,
                  **common_kwargs)
    t_ovlp = bench(run_dispatch_overlap_linear,
                   num_warmup=args.num_warmup, num_iters=args.num_iters,
                   **common_kwargs)

    if rank == 0:
        print("=" * 60)
        print("  FusedDispatch + Linear Overlap Benchmark")
        print("=" * 60)
        print(f"  Tokens       : {args.num_tokens}")
        print(f"  Hidden       : {args.hidden_size}")
        print(f"  Experts      : {num_experts}  (topk={args.topk})")
        print(f"  Linear       : {args.hidden_size} -> {ffn_hidden}")
        print(f"  DeepEP SMs   : {args.num_sms}")
        print(f"  GPUs         : {world_size}")
        print(f"  Warmup/Iters : {args.num_warmup} / {args.num_iters}")
        print("-" * 60)
        print(f"  Sequential  (dispatch ; linear) : {t_seq:.3f} ms")
        print(f"  Overlapped  (dispatch || linear) : {t_ovlp:.3f} ms")
        speedup = t_seq / t_ovlp if t_ovlp > 0 else float("inf")
        saving_pct = (1 - t_ovlp / t_seq) * 100 if t_seq > 0 else 0
        print(f"  Speedup      : {speedup:.2f}x")
        print(f"  Time saved   : {t_seq - t_ovlp:.3f} ms ({saving_pct:.1f}%)")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
