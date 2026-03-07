"""Batch preprocess ShapeNet meshes and segment into patches."""
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.data_prep import load_and_preprocess_mesh
from src.patch_dataset import process_and_save_patches

SHAPENET_CATEGORIES = {
    "chair":    "03001627",
    "table":    "04379243",
    "airplane": "02691156",
    "car":      "02958343",
    "lamp":     "03636649",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="data")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--max_per_category", type=int, default=500)
    args = parser.parse_args()

    shapenet = Path(args.shapenet_root)
    output = Path(args.output_root)
    metadata = []

    for cat_name, cat_id in SHAPENET_CATEGORIES.items():
        cat_dir = shapenet / cat_id
        if not cat_dir.exists():
            print(f"Skipping {cat_name}: {cat_dir} not found")
            continue

        mesh_out = output / "meshes" / cat_name
        patch_out = output / "patches" / cat_name
        mesh_out.mkdir(parents=True, exist_ok=True)

        obj_files = sorted(cat_dir.rglob("*.obj"))[:args.max_per_category]
        print(f"\n[{cat_name}] Processing {len(obj_files)} meshes...")

        for obj_file in tqdm(obj_files, desc=cat_name):
            mesh_id = obj_file.parent.name
            mesh = load_and_preprocess_mesh(str(obj_file), target_faces=args.target_faces)
            if mesh is None:
                continue

            # Save preprocessed mesh
            mesh_file = mesh_out / f"{mesh_id}.obj"
            mesh.export(str(mesh_file))

            # Segment and save patches
            patch_meta = process_and_save_patches(
                str(mesh_file), mesh_id, str(patch_out),
            )
            patch_meta["category"] = cat_name
            metadata.append(patch_meta)

    # Save metadata
    meta_path = output / "patch_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print(f"Total meshes processed: {len(metadata)}")


if __name__ == "__main__":
    main()
