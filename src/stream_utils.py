"""Helpers for streaming dataset processing pipeline."""
import json
from pathlib import Path
from typing import Iterator


class ProgressTracker:
    """Track completed batches for resume-safe processing."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.completed: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                self.completed = json.load(f)

    def is_done(self, batch_id: str) -> bool:
        return batch_id in self.completed

    def mark_done(self, batch_id: str, stats: dict):
        self.completed[batch_id] = stats

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.completed, f, indent=2)


class MetadataCollector:
    """Accumulate per-mesh metadata and persist to JSON."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)

    def add(self, mesh_id: str, entry: dict):
        self.data[mesh_id] = entry

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)


def batch_uids(uids: list, batch_size: int = 500) -> Iterator[list]:
    """Yield successive chunks of uids."""
    for i in range(0, len(uids), batch_size):
        yield uids[i:i + batch_size]


# ShapeNetCore v2: 55 synset IDs → human-readable category names
SHAPENET_SYNSET_MAP = {
    "02691156": "airplane", "02747177": "trash_bin", "02773838": "bag",
    "02801938": "basket", "02808440": "bathtub", "02818832": "bed",
    "02828884": "bench", "02834778": "bicycle", "02843684": "birdhouse",
    "02858304": "boat", "02871439": "bookshelf", "02876657": "bottle",
    "02880940": "bowl", "02924116": "bus", "02933112": "cabinet",
    "02942699": "camera", "02946921": "can", "02954340": "cap",
    "02958343": "car", "02992529": "cellphone", "03001627": "chair",
    "03046257": "clock", "03085013": "keyboard", "03207941": "dishwasher",
    "03211117": "display", "03261776": "earphone", "03325088": "faucet",
    "03337140": "file_cabinet", "03467517": "guitar", "03513137": "helmet",
    "03593526": "jar", "03624134": "knife", "03636649": "lamp",
    "03642806": "laptop", "03691459": "loudspeaker", "03710193": "mailbox",
    "03759954": "microphone", "03761084": "microwave", "03790512": "motorbike",
    "03797390": "mug", "03928116": "piano", "03938244": "pillow",
    "03948459": "pistol", "03991062": "pot", "04004475": "printer",
    "04074963": "remote", "04090263": "rifle", "04099429": "rocket",
    "04225987": "skateboard", "04256520": "sofa", "04330267": "stove",
    "04379243": "table", "04401088": "telephone", "04460130": "tower",
    "04468005": "train", "04530566": "watercraft", "04554684": "washer",
}
