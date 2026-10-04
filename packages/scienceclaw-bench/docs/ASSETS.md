# Model and external-tool assets

The integrated repository carries every wrapper, launcher, configuration, skill,
and provenance record needed to use the external tools. It does not copy model
weights or datasets into Git. Large or licensed binaries stay in the deployment
asset root and are selected by configuration, then verified by their recorded
hash before a tool can be used.

## Manifests and provenance

- `configs/p3_asset_manifest.json` is the pinned inventory for BuildingsBench,
  BEATs, Granite TTM, InnerEye, MASS, and the authorization-blocked TabPFN
  candidate. It records source, revision, expected size/hash, wrapper, smoke
  evidence, and status.
- `configs/formal_toolon_manifest.json` records the required tool references,
  split capacities, and mutually exclusive formal batches. It is a protocol
  manifest, not a weight bundle.
- `reports/toolon_optimization_20261002.md` and the dated `reports/for*_*.md`
  records contain the server-side checkpoint paths and SHA-256 evidence for
  Demucs, SevenNet, CLIP, parser, and other frozen routes.
- `reports/handoff_20261001.md` is the current storage contract: code and
  environments live under `$F`, while data, weights, caches, and run outputs
  live under `$L` on Leonardo.

## Deployment procedure

1. Choose the asset root through `SCIENCECLAW_MODELS` (and
   `SCIENCECLAW_P3_ASSET_ROOT` for the P3 inventory); never place it under the
   Git checkout.
2. Stage only a pinned, license-compatible asset from the manifest. Preserve
   the upstream revision and expected size/hash alongside the deployment log.
3. Run the matching read-only smoke or cache audit under `scripts/leonardo/`
   before exposing the wrapper to an agent. A missing, malformed, unauthorized,
   or partial asset must fail closed.
4. Keep formal tool-on batches separate from visible development diagnostics;
   do not use hidden targets to select a model or generate a skill.

The gateway plugin receives only configured roots and a Python interpreter. It
never accepts a weight path, download URL, credential, or arbitrary command from
the agent request.
