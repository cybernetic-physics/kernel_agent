---
name: modal-b200-snippet-policy
description: Enforce mandatory Modal execution for any Python, CUTLASS, or GPU snippet in lean4real. Use when running quick sanity checks, ad-hoc code snippets, or CUTLASS experiments so execution always goes through tools/modal_python_exec.py on B200 with required env sourcing, sanity prints, and one-retry failure handling.
---

# Modal B200 Snippet Policy

Run all Python/CUTLASS/GPU snippets through Modal B200 only. Do not run local Python snippet execution for these tasks.

## Required policy

1. Source env first:
```bash
set -a; source ../kernel_rl/.env; set +a
```
2. Execute via Modal executor:
```bash
uv run --with modal python tools/modal_python_exec.py --gpu B200 ...
```
3. Allow only these snippet forms:
   - `--code "..."`
   - `--code-file /path/to/snippet.py`
   - stdin/heredoc piped into `tools/modal_python_exec.py`
4. For CUTLASS sanity checks, include:
   - `import cutlass`
   - `import torch`
   - prints for `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)`
5. On failure:
   - show stderr
   - retry once with the exact same command
   - if it fails again, stop and revert the last kernel change
6. Never run local snippet heredocs like:
```bash
python3 - <<'PY'
...
PY
```

## Preferred command wrapper

Use this wrapper script so env/gpu/repo defaults are enforced:

```bash
bash scripts/run_modal_snippet.sh --code "import torch; print(torch.cuda.is_available())"
```

## Command patterns

Run inline code:

```bash
bash scripts/run_modal_snippet.sh \
  --code "import torch, cutlass; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run a code file:

```bash
bash scripts/run_modal_snippet.sh --code-file /tmp/snippet.py
```

Run piped stdin snippet:

```bash
cat /tmp/snippet.py | bash scripts/run_modal_snippet.sh
```

## CUTLASS sanity template

```bash
bash scripts/run_modal_snippet.sh --code "import cutlass, torch; print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## Retry rule

`scripts/run_modal_snippet.sh` already retries once on failure using the same command and surfaces stderr. If the second attempt fails, stop and revert the last kernel edit.
