# Skills Cross-Agent Compatibility (Codex + Claude Code)

This repository keeps one canonical skill set in:
- `skills/codex/<skill-name>/`

To make the same skills discoverable by Claude Code, we mirror them into:
- `.claude/skills/<skill-name>` (symlinks to `skills/codex/*`)

## Why this layout

Claude Code project skills are discovered from `.claude/skills` and each skill directory must contain a `SKILL.md` file with YAML frontmatter.

Codex in this repo already uses `skills/codex/...` as the skill source of truth.

By symlinking `.claude/skills/*` to `skills/codex/*`, we avoid duplicated skill content and keep both tools aligned.

## Source references

- Anthropic Claude Code docs (skills):
  - project skills live under `.claude/skills`
  - each skill requires `SKILL.md` with YAML frontmatter
  - <https://docs.anthropic.com/en/docs/claude-code/skills>
- Anthropic docs note Claude Code skills follow the Agent Skills open standard:
  - <https://docs.anthropic.com/en/docs/claude-code/skills>

## Maintenance

When adding/removing a skill under `skills/codex`, resync Claude links:

```bash
bash tools/sync_claude_skills.sh
```

## Notes

- Keep `SKILL.md` frontmatter minimal and compatible (for example `name`, `description`).
- Prefer relative paths in `SKILL.md`/scripts so symlinked layout still works from `.claude/skills`.
