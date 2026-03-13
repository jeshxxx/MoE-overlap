"""
Test: overlap FusedCombine all-to-all communication with Linear computation.

Typical MoE layer timeline:
    dispatch (a2a) -> expert FFN -> combine (a2a) -> next-layer attention

This script benchmarks overlapping the *combine* phase with an independent
linear layer (simulating next-layer attention projection).

Run (mpirun):
    mpirun --allow-run-as-root -np 8 python3 test_combine_overlap.py
Run (torchrun):
    torchrun --nproc_per_node=8 test_combine_overlap.py
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist

from fused_a2a import (
    fused_dispatch,
    fused_combine,
    dispatch_wait,
    set_deepep_num_sms,
)


# ---------------------------------------------------------------------------
# Distributed setup (mpirun / torchrun / srun)
# ---------------------------------------------------------------------------

def _env_first(*keys, default="0"):
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return default


def setup():
    local_rank = int(_env_first(
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        default="0",
    ))
    rank = int(_env_first(
        "RANK",
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
        "PMI_RANK",
        default="0",
    ))
    world_size = int(_env_first(
        "WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "PMI_SIZE",
        default="1",
    ))

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


# ---------------------------------------------------------------------------
# Prepare dispatch handle (run once, shared by all benchmark iterations)
# ---------------------------------------------------------------------------

def prepare_dispatch(x, token_indices, token_probs, num_experts, group):
    """Run a synchronous dispatch and return (recv_x, handle)."""
    recv_x, _, _, _, handle, event = fused_dispatch(
        x, token_indices, token_probs, num_experts, group,
        async_finish=False,
    )
    dispatch_wait(event)
    return recv_x, handle


# ---------------------------------------------------------------------------
# Sequential vs Overlapped combine
# ---------------------------------------------------------------------------

def run_combine_then_linear(expert_out, group, handle, linear, linear_input):
    """Sequential: synchronous combine, then linear."""
    combined_x, _, event = fused_combine(
        expert_out, group, handle,
        async_finish=False,
    )
    dispatch_wait(event)
    linear_out = linear(linear_input)
    return combined_x, linear_out


def run_combine_overlap_linear(expert_out, group, handle, linear, linear_input):
    """Overlapped: async combine || linear, then sync."""
    combined_x, _, event = fused_combine(
        expert_out, group, handle,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    linear_out = linear(linear_input)
    dispatch_wait(event)
    return combined_x, linear_out


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def bench(fn, num_warmup=5, num_iters=20, **kwargs):
    for _ in range(num_warmup):
        fn(**kwargs)
        torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(num_iters):
        fn(**kwargs)
    end_evt.record()
    torch.cuda.synchronize()

    return start_evt.elapsed_time(end_evt) / num_iters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FusedCombine + Linear overlap")
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

    set_deepep_num_sms(args.num_sms)

    # ---- test tensors ----
    x = torch.randn(args.num_tokens, args.hidden_size,
                     device=device, dtype=torch.bfloat16)
    token_indices = torch.randint(
        0, num_experts, (args.num_tokens, args.topk), device=device)
    token_probs = torch.randn(
        args.num_tokens, args.topk, device=device, dtype=torch.float32
    ).softmax(dim=-1)

    # Independent input for the linear (simulates next-layer attention data)
    linear_input = torch.randn(args.num_tokens, args.hidden_size,
                               device=device, dtype=torch.bfloat16)
    linear = nn.Linear(args.hidden_size, ffn_hidden, bias=False,
                       device=device, dtype=torch.bfloat16)

    group = dist.group.WORLD

    # ---- prepare: run dispatch once to get recv_x and handle ----
    recv_x, handle = prepare_dispatch(
        x, token_indices, token_probs, num_experts, group)

    # Simulate expert FFN output (same shape as recv_x)
    expert_out = torch.randn_like(recv_x)

    common = dict(
        expert_out=expert_out,
        group=group,
        handle=handle,
        linear=linear,
        linear_input=linear_input,
    )

    # ---- benchmark ----
    t_seq = bench(run_combine_then_linear,
                  num_warmup=args.num_warmup, num_iters=args.num_iters,
                  **common)
    t_ovlp = bench(run_combine_overlap_linear,
                   num_warmup=args.num_warmup, num_iters=args.num_iters,
                   **common)

    if rank == 0:
        print("=" * 60)
        print("  FusedCombine + Linear Overlap Benchmark")
        print("=" * 60)
        print(f"  Tokens       : {args.num_tokens}")
        print(f"  Hidden       : {args.hidden_size}")
        print(f"  Experts      : {num_experts}  (topk={args.topk})")
        print(f"  Linear       : {args.hidden_size} -> {ffn_hidden}")
        print(f"  DeepEP SMs   : {args.num_sms}")
        print(f"  GPUs         : {world_size}")
        print(f"  Warmup/Iters : {args.num_warmup} / {args.num_iters}")
        print("-" * 60)
        print(f"  Sequential  (combine ; linear) : {t_seq:.3f} ms")
        print(f"  Overlapped  (combine || linear) : {t_ovlp:.3f} ms")
        speedup = t_seq / t_ovlp if t_ovlp > 0 else float("inf")
        saving_pct = (1 - t_ovlp / t_seq) * 100 if t_seq > 0 else 0
        print(f"  Speedup      : {speedup:.2f}x")
        print(f"  Time saved   : {t_seq - t_ovlp:.3f} ms ({saving_pct:.1f}%)")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
