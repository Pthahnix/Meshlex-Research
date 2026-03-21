# Dataset Pipeline Progress Report — ShapeNet Phase

**Generated:** 2026-03-21 11:18 UTC

## Overall Status
- **Phase**: ShapeNet streaming (Phase D-2)
- **Pipeline PID**: CRASHED (HF rate limit)
- **Categories completed**: 51 / 55 (93%) — 49 OK, 2 errors
- **Blocked by**: HuggingFace 128 commits/hour rate limit

## Stats
- Meshes OK: 30,456
- Meshes fail: 9,861
- Success rate: 75.5%
- Total patches: 4,967,351
- Avg patches/mesh: 163.1

## Disk Usage
- Used: 20GB / 80GB (25%)
- Free: 61GB

## Action Required
- Wait ~1 hour for HF rate limit reset (crashed at ~11:15 UTC)
- Restart at ~12:15 UTC: script auto-skips completed categories via progress.json
- Remaining: table(8509), telephone, tower, train, watercraft, washer (4 categories + table retry)

## Notes
- 2 categories (bicycle, boat) 404 errors — zips not on HF
- Objaverse phase (D-1) completed: 32,136 OK, 4,619,061 patches
