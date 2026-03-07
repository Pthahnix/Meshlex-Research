# Task 4 Validation Report

**Date:** 2026-03-07 08:36:31

## Patch Serialization

| Mesh | Patches | Face Range | NPZ Files | Time |
|------|---------|------------|-----------|------|
| bunny | 29 | [33, 35] | 29 | 0.036s |
| capsule | 13 | [34, 35] | 13 | 0.109s |
| icosphere_fine | 29 | [33, 35] | 29 | 0.037s |
| torus | 29 | [34, 36] | 29 | 0.037s |

**Total patches:** 100

## PatchDataset Loading

- All patches load correctly as PyTorch tensors
- face_features: (80, 15) float32 — padded to MAX_FACES
- edge_index: (2, E) int64 — face adjacency graph
- local_vertices: (60, 3) float32 — padded to MAX_VERTICES

## Visualizations

![Task 4 Summary](task4_summary.png)

![Edge Statistics](task4_edge_stats.png)

## Mesh Previews

### bunny
![bunny preview](bunny_preview.png)

### capsule
![capsule preview](capsule_preview.png)

### icosphere_fine
![icosphere_fine preview](icosphere_fine_preview.png)

### torus
![torus preview](torus_preview.png)

## Conclusion

- Patch serialization produces correct .npz files with all required fields
- PatchDataset loads patches and computes 15-dim face features correctly
- Edge index construction matches expected manifold topology (E ≈ 1.5F)
- Feature padding works correctly for batch training