# AGENTS.md

Instructions for AI coding agents working on this repository.

See [README.md](README.md) for what the project does and [README.dev.md](README.dev.md) for setup, tooling, and code style.

## Checks

Install the git hooks once per clone, before making any change:

```bash
uv run prek install
```

Run these before proposing a change:

```bash
uv run prek run --all-files --stage pre-push
uv run pytest
```

The hooks do not run the tests.

Tests marked `julia` or `gpu` need Julia and a CUDA GPU and are skipped by default. Run them with `uv run pytest -m "julia or gpu"` only when the machine has both.

## AI use disclosure

Every commit made with the help of an AI agent must end with an attribution trailer:

```
Assisted-by: <agent>, <model>
```

For example:

```
Assisted-by: Claude Code, claude-opus-5-5
```

If a commit has no trailer, the contributor must disclose AI use in the pull request description, following the pull request template.
