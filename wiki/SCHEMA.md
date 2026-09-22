# Wiki schema

Maintenance contract for this wiki. Read this before editing any page.

## Layout

```
wiki/
  SCHEMA.md   this contract        (nav file, no frontmatter)
  index.md    catalog of all pages (nav file, no frontmatter)
  log.md      append-only log      (nav file, no frontmatter)
  docs/       topic pages          (concept page, frontmatter required)
```

`docs/` at the **repo root** is a separate, pre-existing reader-facing doc set
(`DESIGN.md`, `EXTENDING.md`, `EVALUATING_POLICIES.md`, `EXPERIMENT_LOG.md`,
`EXPERIMENTS.md`). The wiki does **not** mirror it: `index.md` routes to it, and
`wiki/docs/` covers only what it does not already say.

## Operations

**Ingest** — after a change to behavior, architecture, or conventions:

1. Use the Source→Doc Map to find affected pages.
2. Edit them. Keep prose durable (*why* a thing is so), never run output.
3. Append one `log.md` entry.
4. Update `index.md` only if a page was added or removed.

**Query** — start at `index.md`; it routes to both wiki pages and root `docs/`.

**Lint**

- Every `wiki/docs/` page has a `type:` from the vocabulary.
- Every page appears exactly once in `index.md`.
- No page restates root `docs/` content — link instead.
- No metric tables, run output, or checkpoint scores pasted into a page; those live
  in `runs/` and are regenerated, not documented.
- No machine-specific paths, home directories, or names of other projects.

## Log entry format

```
## [YYYY-MM-DD] {ingest|decision|review} | Title

One to four lines: what changed and why it matters. Link pages as [[page-name]].
```

## `type:` vocabulary

| `type` | Use for |
| --- | --- |
| `convention` | Repo rules, layout, naming, what is safe to change |
| `ops` | How to run, train, evaluate, reproduce |
| `decision` | A durable choice and the reasoning behind it |

## Source→Doc Map

| Source | Page to update |
| --- | --- |
| `src/reservation_pricing/envs/`, `controls/` | root `docs/DESIGN.md` |
| `src/reservation_pricing/demand/` | root `docs/DESIGN.md` |
| `src/reservation_pricing/metrics.py`, `evaluate/` | root `docs/EVALUATING_POLICIES.md`; [[conventions]] for `evaluate/report.py` |
| `src/reservation_pricing/cli.py`, `pyproject.toml` scripts | [[dev-commands]] |
| `configs/*.yaml` | [[conventions]] |
| `artifacts/`, `runs/` | [[conventions]]; root `docs/EXPERIMENTS.md` for a new experiment |
| `scripts/`, `site/`, `.github/` | [[dev-commands]] |
