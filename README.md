# kernel_agents

Workspace for coding agents (ChatGPT/Codex) to develop and evaluate GPU kernels for gpumode/popcorn-style hackathons.

## Layout
- `tools/` local utilities for profiling, Modal execution, PTX/CUBIN inspection, and CUTLASS compile debugging
- `kernels/` place new kernel submissions here
- `artifacts/` run outputs (test/benchmark/leaderboard logs)
- `docs/POPCORN_RUN_GUIDE.md` quick reference for nvfp4_group_gemm run flow

## Setup
Install the local tool environment once:

```bash
cd /path/to/kernel_agents
uv sync
```

## B200 Run Commands (from repo root)
Use these for POPCORN submissions:

```bash
uv run popcorn-cli submit --no-tui --gpu B200 --leaderboard nvfp4_group_gemm --mode test kernels/nvfp4_group_gemm/<kernel>.py
```

```bash
uv run popcorn-cli submit --no-tui --gpu B200 --leaderboard nvfp4_group_gemm --mode benchmark --output artifacts/<kernel>.benchmark.txt kernels/nvfp4_group_gemm/<kernel>.py
```

```bash
uv run popcorn-cli submit --no-tui --gpu B200 --leaderboard nvfp4_group_gemm --mode leaderboard --output artifacts/<kernel>.leaderboard.txt kernels/nvfp4_group_gemm/<kernel>.py
```

## Modal Python/CUTLASS Execution Policy
Any Python/CUTLASS/GPU snippet should run on Modal B200:

```bash
set -a; source .env; set +a
# deploy once (or after tools/modal_tools_app.py changes)
uv run --with modal modal deploy tools/modal_tools_app.py
uv run --with modal python tools/modal_python_exec.py --gpu B200 --mount-repo --code "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Useful tools
- `tools/modal_python_exec.py`: run arbitrary Python on B200 Modal with repo mount support.
- `tools/profile_kernel_torch_modal.py`: collect torch profiler traces on Modal.
- `tools/dump_ptx_modal.py`: extract PTX for a kernel on Modal.
- `tools/disassemble_cubin_modal.py`: disassemble CUBIN/SASS on Modal.
- `tools/debug_cutlass_compile_modal.py`: gather detailed CUTLASS compile diagnostics on Modal.

## Skills Compatibility (Codex + Claude Code)
- Canonical skills live in `skills/codex/`.
- Claude Code project discovery uses `.claude/skills/`.
- This repo keeps `.claude/skills/*` as symlinks to `skills/codex/*` so both agents use the same skill definitions.
- Resync links after skill changes:

```bash
bash tools/sync_claude_skills.sh
```

See `docs/SKILLS_CROSS_AGENT_COMPAT.md` for details and references.
