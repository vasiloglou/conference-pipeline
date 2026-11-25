# Conference Workspaces

This directory holds one workspace per conference.

## Model

The repo is organized as one shared platform with many conference workspaces:

- shared platform code lives in `src/`, root `scripts/`, `tests/`, and `docs/`
- each conference gets its own workspace under `conferences/<slug>/`

Current workspaces:

- `neurips2025/`
- `iclr2025/`

Starter template:

- `TEMPLATE/`

## Naming

Use lowercase slug-plus-year directory names:

- `neurips2025`
- `icml2026`
- `icml2026`
- `mlsys2027`

Conference-specific GSD milestone display names use
`{platform_version}-{conference name}`:

- `v1.0-NeurIPS 2025`
- `v1.2-ICLR 2025`

Keep archived GSD filenames stable unless a future tooling change explicitly
supports safe archive renames.

## GSD Context Packets

Every real conference milestone must have a formal planning context file under
`.planning/milestones/`. The packet declares the local milestone identity,
source URLs, auth notes, analysis profile, supported stages, proof status,
canonical local docs, and known limitations.

Root docs stay platform-general. Put conference-specific topic inventories and
proof receipts in the planning-owned context packet. Workspace READMEs can link
to that packet.

For a new conference analysis, the normal path is:

1. Create a PRD-style conference brief.
2. Run `$plan-conference-analysis` to generate the analysis specification and
   workspace scaffold.
3. Review and finalize the generated spec after its audits pass.
4. Run `$plan-conference-production` from the finalized spec to generate
   live-only production phases or a production milestone.

Do not treat the analysis-spec milestone as live production completion. It is a
planning proof and handoff. Production work should usually use a new milestone
or workstream so live receipts, provider costs, backup checkpoints, and human
review evidence remain separate from dry-run/mock planning evidence.

## Expected Workspace Layout

Each workspace should use the shared lane-based storage convention by default.
The stable top-level lanes are:

- `analysis/` - private analysis notes and intermediate writeups
- `decks/` - curated presentation decks and visual release outputs
- `audio/` - podcast, narration, and other release-oriented audio outputs
- `exports/` - private export bundles such as deep-dive or exhaustive extraction output
- `agent-work/` - private agent sidecar work and generated working material
- `cache/` - rebuildable local indexes, experiments, and cache state
- `legacy/` - archived material retained for traceability but not active work

Each workspace should also contain the support material that makes the storage
contract durable:

- `README.md`
- `config.yaml`
- `storage/`
- `assets/`
- `scripts/`

`config.yaml` should declare `storage.manifest_path:
"storage/storage_manifest.json"` and a `workspace_layout` block. The manifest at
`storage/storage_manifest.json` defines which artifact families are public,
private backup material, git-tracked support material, or rebuildable cache.

Conference-local subfolders are allowed beneath the stable lanes. For example,
one conference can split `decks/` by topic while another splits it by audience,
as long as the manifest paths remain accurate.

## Rule Of Thumb

Add a new conference here by default. Do not create a separate top-level project unless the conference stops fitting the shared platform assumptions.
