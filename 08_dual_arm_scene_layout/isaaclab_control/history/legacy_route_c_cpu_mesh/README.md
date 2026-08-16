# Legacy Route-C CPU IK and mesh collision

Archived on 2026-08-16 after the formal runtime switched to `core/` Route-C V2.

This directory preserves the former SciPy/Pinocchio IK, coarse reachability,
complete-mesh/table joint-path collision, and their orchestration scripts for
diagnosis and provenance only. They are not imported or called by the production
runtime. A coarse FAIL from these scripts is not an exact physical-unreachable
certificate.
