# Battery baseline checksum audit

Audit date: 2026-08-14.

Original `SHA256SUMS` verification: **8 PASS / 2 FAIL**.

| File | Recorded SHA256 | Current SHA256 | File mtime (local) |
|---|---|---|---|
| `task/final_waypoints.json` | `364db4af54ac49eaf3a4239fcb4216abb52928168fd111252d1c3502cba29d8c` | `e11f56e9bac51d2ccba5e1feac4c47d073661637989ae804178f87be843973ba` | 2026-08-11 23:14:38 +08:00 |
| `runtime/common_import.py` | `6beba14572193b01d488470fbf36a52975db865d8d2f87376435d31b163b7cca` | `0cf11ba505b1cb41edd42b4897c8d1612777df2a4aab1a16c933587d2f4112f6` | 2026-08-11 23:14:38 +08:00 |

The baseline `SHA256SUMS` mtime is 2026-08-11 18:58:03 +08:00. Both mismatched files were modified later that evening, before Phase 0-2. No file inside the immutable Battery baseline was changed during this reorganization.

Classification: **VERIFIED_PHYSICAL** + **PRE_EXISTING_INTEGRITY_EXCEPTION**. The observed values for all ten tracked files are recorded in `observed_sha256_20260814.txt`; the original checksum remains untouched.
