import numpy as np
import pytest
from src.graph_bpe import GraphBPE, BPEVocabulary
from src.dual_graph import DualGraph


def _make_dual_graph(n_nodes=6, n_edges=10, max_label=4, seed=42):
    """Create a small synthetic dual graph."""
    rng = np.random.RandomState(seed)
    node_labels = rng.randint(0, max_label, n_nodes)
    # Random edges (undirected -> bidirectional)
    pairs = set()
    while len(pairs) < n_edges:
        u, v = sorted(rng.randint(0, n_nodes, 2))
        if u != v:
            pairs.add((u, v))
    pairs = list(pairs)
    src = np.array([p[0] for p in pairs] + [p[1] for p in pairs])
    dst = np.array([p[1] for p in pairs] + [p[0] for p in pairs])
    edge_labels = rng.randint(0, 3, len(src))
    return DualGraph(
        n_nodes=n_nodes,
        node_labels=node_labels,
        edge_src=src,
        edge_dst=dst,
        edge_labels=edge_labels,
    )


def test_bpe_vocabulary_grows():
    """After training, vocabulary should be larger than base alphabet."""
    graphs = [_make_dual_graph(seed=i) for i in range(5)]
    bpe = GraphBPE(target_vocab_size=10)
    vocab = bpe.train(graphs)
    assert len(vocab.symbols) > 4  # base alphabet has 4 labels


def test_bpe_encode_produces_patches():
    """Encoding a graph should produce a list of face groups."""
    graphs = [_make_dual_graph(n_nodes=20, n_edges=30, seed=i) for i in range(10)]
    bpe = GraphBPE(target_vocab_size=15)
    vocab = bpe.train(graphs)
    patches = bpe.encode(graphs[0], vocab)
    # Each patch is a list of face indices
    assert len(patches) > 0
    # All faces should be covered
    all_faces = set()
    for p in patches:
        all_faces.update(p.face_indices)
    assert all_faces == set(range(graphs[0].n_nodes))


def test_bpe_deterministic():
    """Same input should produce same output."""
    graphs = [_make_dual_graph(seed=i) for i in range(5)]
    bpe1 = GraphBPE(target_vocab_size=10)
    vocab1 = bpe1.train(graphs)
    bpe2 = GraphBPE(target_vocab_size=10)
    vocab2 = bpe2.train(graphs)
    assert len(vocab1.symbols) == len(vocab2.symbols)


def test_bpe_merge_count():
    """Number of merges should equal vocab_size - base_alphabet_size."""
    graphs = [_make_dual_graph(seed=i) for i in range(5)]
    bpe = GraphBPE(target_vocab_size=10)
    vocab = bpe.train(graphs)
    assert len(vocab.merge_rules) == len(vocab.symbols) - vocab.base_alphabet_size
