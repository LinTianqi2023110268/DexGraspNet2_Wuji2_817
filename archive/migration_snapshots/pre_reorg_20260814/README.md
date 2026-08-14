# Pre-reorganization snapshot — 2026-08-14

This directory is the Phase 0 rollback and audit record taken before documentation cleanup and verified-baseline indexing.

No project file was moved or deleted while producing it. The large training datasets were inventoried in place and were not copied into this directory.

Key files:

- `full_file_manifest.tsv`: complete pre-reorganization file inventory, including a supplementary pass for 1,382 pre-existing symlink names.
- `critical_sha256.txt`: hashes for critical datasets, checkpoints, assets, results, and entry points.
- `case_inventory.tsv`: metadata-driven A/B/C case classification.
- `06_case_reduction_plan.tsv` and `08_reorg_plan.tsv`: future plans only; no listed move was executed.
- `delete_candidates.tsv`: review list only; no listed deletion was executed.
- `pre_reorg_text_backup.tar.gz`: recoverable backup of the text/configuration layer before Phase 1 edits.

The snapshot metadata records 298,860 regular files and 338,124,520,312 bytes in the project before reorganization, excluding this snapshot directory itself.
