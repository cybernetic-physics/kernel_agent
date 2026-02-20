# GPU Mode Run Guide (Test / Benchmark / Leaderboard)

This guide shows how to run `popcorn-cli` for this kernel from repo root:

```bash
cd /Users/cuboniks/Projects/kernel_projects/lean4real
export PATH="$HOME/.local/bin:$PATH"
```

## 1) Prerequisites

```bash
which popcorn-cli
cat ~/.popcorn.yaml
```

If either fails:

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn-cli register github
```

## 2) Test (Correctness)

Use this before any benchmark/leaderboard submission:

```bash
popcorn-cli submit --no-tui \
  --gpu B200 \
  --leaderboard nvfp4_group_gemm \
  --mode test \
  kernels/nvfp4_group_gemm/wagmiv67.py
```

## 3) Benchmark (Timing Only)

```bash
popcorn-cli submit --no-tui \
  --output kernels/nvfp4_group_gemm/wagmiv67.benchmark.txt \
  --gpu B200 \
  --leaderboard nvfp4_group_gemm \
  --mode benchmark \
  kernels/nvfp4_group_gemm/wagmiv67.py
```

## 4) Leaderboard (Ranked)

```bash
popcorn-cli submit --no-tui \
  --output kernels/nvfp4_group_gemm/wagmiv67.leaderboard.txt \
  --gpu B200 \
  --leaderboard nvfp4_group_gemm \
  --mode leaderboard \
  kernels/nvfp4_group_gemm/wagmiv67.py
```

## 5) Compare Versions Quickly

Replace the file path with any version:

- `kernels/nvfp4_group_gemm/wagmi_v5.py`
- `kernels/nvfp4_group_gemm/wagmi_v6.py`
- `kernels/nvfp4_group_gemm/wagmiv67.py`

Example loop:

```bash
for f in wagmi_v5.py wagmi_v6.py wagmiv67.py; do
  popcorn-cli submit --no-tui --gpu B200 --leaderboard nvfp4_group_gemm --mode benchmark "kernels/nvfp4_group_gemm/$f"
done
```

## 6) Notes

- Always run `--mode test` first.
- Use `--no-tui` in non-interactive shells.
- `--gpu B200` is explicit; `--gpu NVIDIA` may route differently.
