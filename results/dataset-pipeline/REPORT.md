# Dataset Pipeline Progress Report — ShapeNet Phase

**Generated:** 2026-03-21 06:18 UTC

## Overall Status
- **Phase**: ShapeNet streaming (Phase D-2)
- **Pipeline PID**: 223161 (running)
- **Categories completed**: 14 / 55 (25%) — 12 OK, 2 errors
- **Currently processing**: cabinet (02933112)

## Stats
- Meshes OK: 6,483
- Meshes fail: 3,151
- Success rate: 67.3%
- Total patches: 1,204,789
- Avg patches/mesh: 185.8

## Per-Category Breakdown
| Category | Synset | OK | Fail | Patches |
|----------|--------|----|------|---------|
| airplane | 02691156 | 1,957 | 2,088 | 549,219 |
| trash_bin | 02747177 | 302 | 41 | 60,980 |
| bag | 02773838 | 79 | 4 | 14,038 |
| basket | 02801938 | 88 | 25 | 10,941 |
| bathtub | 02808440 | 803 | 53 | 87,883 |
| bed | 02818832 | 213 | 20 | 33,632 |
| bench | 02828884 | 1,503 | 310 | 251,431 |
| bicycle | 02834778 | — | — | ERROR: 404 |
| birdhouse | 02843684 | 72 | 1 | 3,724 |
| boat | 02858304 | — | — | ERROR: 404 |
| bookshelf | 02871439 | 389 | 63 | 29,181 |
| bottle | 02876657 | 491 | 7 | 36,917 |
| bowl | 02880940 | 182 | 4 | 11,709 |
| bus | 02924116 | 404 | 535 | 115,134 |

## Disk Usage
- Used: 15GB / 80GB (19%)
- Free: 66GB

## Latest Log
```
[cabinet] Writing sub-batch (61234 patches)...
```

## Notes
- 2 categories (bicycle, boat) returned 404 — zips not on HF ShapeNet repo.
- bus had high fail rate (57%) similar to airplane — complex vehicle meshes.
- Pipeline started at 04:35 UTC (~103 min elapsed).
- Objaverse phase (D-1) completed earlier: 93 batches, 32,136 OK, 4,619,061 patches.
