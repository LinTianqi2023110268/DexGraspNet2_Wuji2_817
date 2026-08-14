# Git Snapshot Report

Snapshot date: 2026-08-15  
Snapshot identity: `pre-final-cleanup-2026-08-14`

## Main repository

- Repository: `LinTianqi2023110268/DexGraspNet2_Wuji2`
- URL: `https://github.com/LinTianqi2023110268/DexGraspNet2_Wuji2`
- Snapshot commit: `da928ff1aed7b50d1b992b4ac2f19b2639a00768`
- Annotated tag: `pre-final-cleanup-2026-08-14`
- Visibility: `PRIVATE` (verified through the GitHub API).

## Dependency repositories

### `wuji-description`

- Private repository:
  `https://github.com/LinTianqi2023110268/DexGraspNet2_Wuji2-wuji-description`
- Project branch: `dexgraspnet2-wuji2`
- Project commit: `96cb3238d4bebd23403aecee02ba731d52524faa`
- Original upstream:
  `https://github.com/wuji-technology/wuji-description.git`
- Upstream base commit: `8271644a78d69ed9a4adcf9165d882c64ad33dfa`

### `wuji-retargeting`

- Private repository:
  `https://github.com/LinTianqi2023110268/DexGraspNet2_Wuji2-wuji-retargeting`
- Project branch: `dexgraspnet2-wuji2`
- Project commit: `52ed22779915ca36f7c9a736eea6828a342d1c36`
- Original upstream:
  `https://github.com/wuji-technology/wuji-retargeting.git`
- Upstream base commit: `2918c60643cca3482ffa2d14d1f7fece1d9d7db9`

## Pinned submodules

- `01_environment/vendor/wuji-description`
  -> `96cb3238d4bebd23403aecee02ba731d52524faa`
- `01_environment/vendor/wuji-retargeting`
  -> `52ed22779915ca36f7c9a736eea6828a342d1c36`

The gitlinks above pin exact commits. They do not implicitly follow the newest
upstream `main` branch.

## Submission scope

- Tracked entry count: 1,827
- Cached uncompressed blob size before the report was added: 237.65 MiB
- Main `.git` working-copy size after the snapshot push: 93 MiB
  (`git count-objects`: 90.66 MiB loose-object payload, zero garbage).
- Primary ignored data/environment directories: 336,447,580,266 bytes
  (313.34 GiB). This is a conservative subtotal, not a claim that every
  ignored cache in the working tree was enumerated.
- Large-file audit: PASS; 213/213 files over 95 MiB are ignored.
- Secret audit: PASS; no real credential signature was found.
- `xwechat` / `wxid` path audit: PASS; zero tracked occurrences remain.
- Remaining `/home/lin` files in staged content:
  - historical provenance/generated results: 85 files
  - active runtime code or active documentation: 57 files
- No global `/home/lin` replacement was performed.

## Explicitly not uploaded

- `01_environment/conda/`
- `02_training_dataset/data/` (formal training data, about 310+ GiB)
- `02_training_dataset/assets/wuji2_factory/`
- model checkpoints (`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`)
- large physical traces and replay arrays
- video/frame caches and generated presentation outputs
- regenerable bulk case archives

No ignored local file was deleted or moved.

## Personal-path handling

Two temporary WeChat download paths were removed before the dependency commit:

1. `wuji-description/dual_arm/README.md`
   -> `<LOCAL_DOWNLOAD_SOURCE>/dual_arm.zip`
2. `wuji-description/dual_arm_right_wuji2/config/assembly_spec.json`
   -> `<LOCAL_DOWNLOAD_SOURCE>/dual_arm_wuji_assembly.zip`

The filenames, SHA-256 provenance and assembly semantics were preserved.

## Verified project state

- `tools_validate.py`: PASS under the `graspnet2.0` environment.
- Route A: Ashtray source462 verified.
- Route B: Battery candidate49 physical PASS with the pre-existing 8/10
  checksum condition recorded as `PRE_EXISTING_INTEGRITY_EXCEPTION`.
- Route C: dog candidate3800 full pick-place verified, 19/19 SHA PASS.
- No Isaac Sim, training, inference or dataset generation was run during the
  Git snapshot operation.

## Recovery

To inspect the frozen snapshot after cloning:

```bash
git checkout pre-final-cleanup-2026-08-14
git submodule update --init --recursive
```

Do not execute `git checkout` in the current working project merely to verify
this documentation; the command is recorded for future recovery only.

## Recursive clone verification

- Temporary clone path:
  `/tmp/DexGraspNet2_Wuji2_clone_verify_5dSe07`
- Main clone: PASS at
  `da928ff1aed7b50d1b992b4ac2f19b2639a00768`.
- `wuji-description`: PASS at pinned commit
  `96cb3238d4bebd23403aecee02ba731d52524faa`.
- `wuji-retargeting`: PASS at pinned commit
  `52ed22779915ca36f7c9a736eea6828a342d1c36`.
- Nested upstream submodules required by `wuji-retargeting`: PASS.
- Expected project documents/configuration: PASS.
- Excluded training data and Conda environment absent from clone: PASS.
