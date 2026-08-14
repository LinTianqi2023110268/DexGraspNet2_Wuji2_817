# B verified LEAP-to-Wuji2

Status: **VERIFIED_PHYSICAL** + **PRE_EXISTING_INTEGRITY_EXCEPTION**.

Canonical immutable case: `../../06_leap_to_wuji2_final_pipeline/04_verified_baseline/scene0001_view0001_official_rank0/`.

The existing baseline was not copied or modified during Phase 0-2.

Checksum audit on 2026-08-14 found that 8 of 10 entries pass. Two files (`task/final_waypoints.json` and `runtime/common_import.py`) have modification times later than the original `SHA256SUMS` and do not match it. This drift predates reorganization. The physical Battery candidate49 result remains verified, but the frozen directory is not bitwise-clean.

See `checksum_audit.md` and `observed_sha256_20260814.txt`. Never rewrite the old checksum. A future bitwise-clean baseline must be regenerated and reverified under a new directory.
