"""Graph BPE: Byte-Pair Encoding on labeled dual graphs.

Extends standard BPE from sequences to graphs. A "bigram" is a pair of
adjacent nodes (u, v) characterized by the triple (label_u, edge_label, label_v).
Merge = contract the most frequent bigram across all training graphs.

Reference: Spec Section 3.2, Graph Tokenization (Guo et al., 2026).
"""
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from src.dual_graph import DualGraph


@dataclass
class BPEPatch:
    """A group of faces produced by BPE encoding."""
    face_indices: list[int]
    token_id: int


@dataclass
class BPEVocabulary:
    """Learned BPE vocabulary."""
    symbols: list[str]                     # All symbol names (base + merged)
    merge_rules: list[tuple[str, str, str]]  # (label_u, edge_label, label_v) merge history
    base_alphabet_size: int


class GraphBPE:
    """Graph BPE learner and encoder."""

    def __init__(self, target_vocab_size: int = 2000):
        self.target_vocab_size = target_vocab_size

    def train(self, graphs: list[DualGraph]) -> BPEVocabulary:
        """Learn BPE vocabulary from a list of labeled dual graphs.

        Algorithm:
        1. Initialize vocabulary = all unique node labels (base alphabet)
        2. Count bigram frequencies across all graphs
        3. Merge most frequent bigram (greedy, ID-ordered)
        4. Repeat until vocab reaches target size

        Args:
            graphs: List of DualGraph objects

        Returns:
            BPEVocabulary with symbols and merge rules
        """
        # Work on mutable copies
        work_graphs = [self._to_mutable(g) for g in graphs]

        # Base alphabet: all unique node labels
        all_labels = set()
        for g in work_graphs:
            all_labels.update(str(l) for l in g["node_labels"])
        symbols = sorted(all_labels)
        base_size = len(symbols)
        merge_rules = []

        n_merges = self.target_vocab_size - base_size
        for step in range(n_merges):
            # Count bigrams
            bigram_counts = self._count_bigrams(work_graphs)
            if not bigram_counts:
                break

            # Most frequent bigram
            best_bigram = max(bigram_counts, key=bigram_counts.get)
            lu, le, lv = best_bigram

            # New merged symbol
            new_symbol = f"M{base_size + step}_{lu}_{le}_{lv}"
            symbols.append(new_symbol)
            merge_rules.append((lu, le, lv))

            # Apply merge to all graphs
            for g in work_graphs:
                self._apply_merge(g, lu, le, lv, new_symbol)

        return BPEVocabulary(
            symbols=symbols,
            merge_rules=merge_rules,
            base_alphabet_size=base_size,
        )

    def encode(self, graph: DualGraph, vocab: BPEVocabulary) -> list[BPEPatch]:
        """Apply learned merges to a graph, returning BPE patches.

        Each resulting node (after all merges) = one patch = group of original faces.

        Args:
            graph: DualGraph to encode
            vocab: Learned BPEVocabulary

        Returns:
            List of BPEPatch, each containing face indices and token ID.
        """
        g = self._to_mutable(graph)

        # Apply merge rules in order
        for i, (lu, le, lv) in enumerate(vocab.merge_rules):
            new_symbol = vocab.symbols[vocab.base_alphabet_size + i]
            self._apply_merge(g, lu, le, lv, new_symbol)

        # Build patches from remaining nodes
        patches = []
        symbol_to_id = {s: i for i, s in enumerate(vocab.symbols)}
        for node_id in range(len(g["node_labels"])):
            if g["alive"][node_id]:
                label = g["node_labels"][node_id]
                token_id = symbol_to_id.get(label, -1)
                patches.append(BPEPatch(
                    face_indices=sorted(g["face_groups"][node_id]),
                    token_id=token_id,
                ))

        return patches

    def _to_mutable(self, graph: DualGraph) -> dict:
        """Convert DualGraph to mutable working representation."""
        n = graph.n_nodes
        node_labels = [str(l) for l in graph.node_labels]
        face_groups = [[i] for i in range(n)]
        alive = [True] * n

        # Adjacency as dict of dicts: adj[u][v] = edge_label
        adj = {i: {} for i in range(n)}
        for src, dst, el in zip(graph.edge_src, graph.edge_dst, graph.edge_labels):
            adj[int(src)][int(dst)] = str(el)

        return {
            "node_labels": node_labels,
            "face_groups": face_groups,
            "alive": alive,
            "adj": adj,
            "next_id": n,
        }

    def _count_bigrams(self, graphs: list[dict]) -> Counter:
        """Count (label_u, edge_label, label_v) bigram frequencies."""
        counts = Counter()
        for g in graphs:
            for u in range(len(g["alive"])):
                if not g["alive"][u]:
                    continue
                for v, el in g["adj"][u].items():
                    if not g["alive"][v]:
                        continue
                    if u >= v:
                        continue  # count each undirected edge once
                    lu = g["node_labels"][u]
                    lv = g["node_labels"][v]
                    # Canonical ordering: (min_label, edge, max_label)
                    if lu <= lv:
                        bigram = (lu, el, lv)
                    else:
                        bigram = (lv, el, lu)
                    counts[bigram] += 1
        return counts

    def _apply_merge(self, g: dict, lu: str, le: str, lv: str, new_symbol: str):
        """Merge all matching bigram pairs in a graph (greedy, ID-ordered).

        For each matched (u, v): contract into u, mark v dead, inherit v's edges.
        """
        merged_this_round = set()

        for u in range(len(g["alive"])):
            if not g["alive"][u] or u in merged_this_round:
                continue
            for v in sorted(g["adj"][u].keys()):
                if not g["alive"][v] or v in merged_this_round:
                    continue
                el = g["adj"][u].get(v)
                if el is None:
                    continue

                label_u = g["node_labels"][u]
                label_v = g["node_labels"][v]

                # Check match (both orderings)
                match = False
                if label_u == lu and el == le and label_v == lv:
                    match = True
                elif label_v == lu and el == le and label_u == lv:
                    match = True

                if not match:
                    continue

                # Merge v into u
                g["node_labels"][u] = new_symbol
                g["face_groups"][u].extend(g["face_groups"][v])
                g["alive"][v] = False
                merged_this_round.add(u)
                merged_this_round.add(v)

                # Inherit v's edges (skip u-v edge)
                for w, w_el in g["adj"][v].items():
                    if w == u or not g["alive"][w]:
                        continue
                    # Add edge u-w (keep existing if present)
                    if w not in g["adj"][u]:
                        g["adj"][u][w] = w_el
                        g["adj"][w][u] = w_el

                # Remove v from all adjacency
                for w in list(g["adj"][v].keys()):
                    if w in g["adj"] and v in g["adj"][w]:
                        del g["adj"][w][v]
                if v in g["adj"][u]:
                    del g["adj"][u][v]
                g["adj"][v] = {}

                break  # u is done for this round
