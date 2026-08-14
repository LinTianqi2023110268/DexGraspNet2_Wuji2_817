# Dependency Git Audit

Audit date: 2026-08-14

This file records the two upstream repositories embedded under
`01_environment/vendor/` before they are registered as Git submodules. The
official upstream history is retained; project-specific work is committed on
the dedicated `dexgraspnet2-wuji2` branch in each dependency.

## `wuji-description`

- Path: `01_environment/vendor/wuji-description`
- Original upstream URL: `https://github.com/wuji-technology/wuji-description.git`
- Original remote name at audit time: `origin`
- Upstream branch: `main`
- Upstream commit used as the base: `8271644a78d69ed9a4adcf9165d882c64ad33dfa`
- Upstream tag at the base: `v2026.8.3`
- Local project additions found before commit:
  - `dual_arm/`
  - `dual_arm_right_wuji2/`
- Personal download paths removed before commit:
  - `dual_arm/README.md`: source path changed to
    `<LOCAL_DOWNLOAD_SOURCE>/dual_arm.zip`
  - `dual_arm_right_wuji2/config/assembly_spec.json`: reference-only archive
    changed to `<LOCAL_DOWNLOAD_SOURCE>/dual_arm_wuji_assembly.zip`
- Project branch: `dexgraspnet2-wuji2`
- Project commit: `96cb3238d4bebd23403aecee02ba731d52524faa`

## `wuji-retargeting`

- Path: `01_environment/vendor/wuji-retargeting`
- Original upstream URL: `https://github.com/wuji-technology/wuji-retargeting.git`
- Original remote name at audit time: `origin`
- Upstream branch: `main`
- Upstream commit used as the base: `2918c60643cca3482ffa2d14d1f7fece1d9d7db9`
- Upstream tag at the base: `v2026.8.3`
- Local project modification found before commit:
  - `wuji_retargeting/robot.py` supports both Pinocchio frame fields
    `parentJoint` and `parent` without changing the retargeting mathematics.
- Project branch: `dexgraspnet2-wuji2`
- Project commit: `52ed22779915ca36f7c9a736eea6828a342d1c36`

## Remote policy

For both dependencies:

- `upstream` preserves the official Wuji Technology repository.
- `origin` will point to the user's private GitHub repository.
- The main project will pin the exact project commit as a submodule gitlink.
- Neither dependency follows the latest upstream branch implicitly.
