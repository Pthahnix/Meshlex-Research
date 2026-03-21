# Dataset Pipeline Progress Report — ShapeNet Phase

**Generated:** 2026-03-21 08:03 UTC

## Overall Status
- **Phase**: ShapeNet streaming (Phase D-2)
- **Pipeline PID**: 223161 (running)
- **Categories completed**: 20 / 55 (36%) — 18 OK, 2 errors
- **Currently processing**: chair (03001627) — 6778 models (largest category)

## Stats
- Meshes OK: 9,486
- Meshes fail: 6,341
- Success rate: 60.0%
- Total patches: 1,783,888
- Avg patches/mesh: 188.1

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
| cabinet | 02933112 | 1,467 | 104 | 173,482 |
| camera | 02942699 | 99 | 14 | 15,777 |
| can | 02946921 | 104 | 4 | 8,203 |
| cap | 02954340 | 55 | 1 | 5,808 |
| car | 02958343 | 492 | 3,022 | 254,232 |
| cellphone | 02992529 | 786 | 45 | 121,597 |

## Disk Usage
- Used: 21GB / 80GB (27%)
- Free: 60GB

## Latest Log
```
[chair] Downloaded + extracted in 59s
[chair] Found 6778 models
(sub-batches uploading to HF)
```

## Timing
- Pipeline started: 04:35 UTC (~3.5h elapsed)
- Processing rate: ~80 meshes/min
- Estimated remaining: ~5-6 hours (table 8509, chair 6778, sofa 3173, etc.)
- 2 categories (bicycle, boat) 404 errors — zips not on HF
- car had 86% fail rate (complex vehicle meshes)
- Objaverse phase (D-1) completed: 32,136 OK, 4,619,061 patches
