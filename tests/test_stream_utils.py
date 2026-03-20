"""Tests for streaming pipeline helpers."""
import json
import pytest


def test_progress_tracker_save_load(tmp_path):
    """ProgressTracker should persist completed batches across restarts."""
    from src.stream_utils import ProgressTracker
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    assert not tracker.is_done("batch_000")
    tracker.mark_done("batch_000", {"meshes": 450, "patches": 12000})
    tracker.save()
    tracker2 = ProgressTracker(str(tmp_path / "progress.json"))
    assert tracker2.is_done("batch_000")
    assert not tracker2.is_done("batch_001")


def test_metadata_collector_accumulate(tmp_path):
    """MetadataCollector should accumulate entries and save to JSON."""
    from src.stream_utils import MetadataCollector
    collector = MetadataCollector(str(tmp_path / "metadata.json"))
    collector.add("mesh_001", {"category": "chair", "source": "objaverse"})
    collector.add("mesh_002", {"category": "table", "source": "shapenet"})
    collector.save()
    with open(tmp_path / "metadata.json") as f:
        data = json.load(f)
    assert "mesh_001" in data
    assert data["mesh_002"]["source"] == "shapenet"


def test_metadata_collector_resume(tmp_path):
    """MetadataCollector should load existing entries on init."""
    from src.stream_utils import MetadataCollector
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"existing": {"category": "lamp"}}, f)
    collector = MetadataCollector(str(tmp_path / "metadata.json"))
    assert "existing" in collector.data
    collector.add("new_mesh", {"category": "car"})
    assert len(collector.data) == 2


def test_shapenet_synset_to_category():
    """Should map synset IDs to human-readable category names."""
    from src.stream_utils import SHAPENET_SYNSET_MAP
    assert SHAPENET_SYNSET_MAP["03001627"] == "chair"
    assert SHAPENET_SYNSET_MAP["02691156"] == "airplane"
    assert len(SHAPENET_SYNSET_MAP) == 57


def test_batch_uids():
    """batch_uids should split a list into chunks of given size."""
    from src.stream_utils import batch_uids
    uids = list(range(1250))
    batches = list(batch_uids(uids, batch_size=500))
    assert len(batches) == 3
    assert len(batches[0]) == 500
    assert len(batches[2]) == 250
