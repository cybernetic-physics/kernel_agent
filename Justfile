set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Launch live worker/reviewer loop using the afterhours optimization prompt.
# Accepts key=value overrides after recipe name, e.g.:
# just loop-afterhours target_threshold=9.5 max_iterations=600
loop-afterhours *opts:
  python3 tools/run_loop_afterhours.py {{opts}}

# Monitor an existing loop state directory without launching a new run.
loop-tui state_root="artifacts/loop_coordinator":
  python3 tools/loop_tui.py --state-root "{{state_root}}"
