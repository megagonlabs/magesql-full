import copy
import random
import time

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from gnn_contrastive_learning.sql_to_graph import SQL2GraphWithFeatures
from gnn_contrastive_learning.sql_graph_augmenter import SQLGraphAugmenter


class GCLDataset(Dataset):
    """
    Dataset for Graph Contrastive Learning.
    Precomputes and stores networkx graphs, dynamically converts them to PyG graphs during retrieval.
    """
    def __init__(self, records, sql_to_graph_with_features, sql_graph_augmenter, num_positives=2, num_negatives=2, num_augmenters=1):
        """
        :param records: List of dataset records, each containing 'query', 'db_id', and precomputed 'graph'.
        :param sql_to_graph_with_features: Instance of SQL2GraphWithFeatures for PyG conversion.
        :param sql_graph_augmenter: Instance of SQLGraphAugmenter for augmenting graphs.
        :param num_positives: Number of positive samples per anchor.
        :param num_negatives: Number of negative samples per anchor.
        """
        self.records = self._precompute_graphs(records, sql_to_graph_with_features)
        self.sql_to_graph_with_features = sql_to_graph_with_features
        self.sql_graph_augmenter = sql_graph_augmenter
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.num_augmenters = num_augmenters

    def _precompute_graphs(self, records, sql_to_graph_with_features):
        """
        Precompute and store networkx graphs for all records.
        :param records: List of dataset records with SQL queries.
        :param sql_to_graph_with_features: Instance of SQL2GraphWithFeatures for graph conversion.
        :return: Updated records with precomputed networkx graphs.
        """
        for record in records:
            sql_query = record["query"]
            record["graph"] = sql_to_graph_with_features.sql_to_graph.sql2networkx(sql_query)
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        """
        Fetch an anchor graph and its augmented positives and negatives.
        :param idx: Index of the anchor record.
        :return: Tuple (anchor_graph, positive_graphs, negative_graphs).
        """
        record = self.records[idx]
        anchor_nx_graph = record["graph"]
        db_id = record["db_id"]

        # Step 1: Convert the anchor networkx graph to PyG
        start_anchor_time = time.time()
        anchor_graph = self.sql_to_graph_with_features.convert_to_pyg(anchor_nx_graph)
        anchor_time = time.time() - start_anchor_time

        # Step 2: Generate positives
        start_positive_time = time.time()
        positive_graphs = []
        for _ in range(self.num_positives):
            augmented_nx_graph = self.sql_graph_augmenter.generate_positive_instance_with_args(
                anchor_nx_graph, db_id=db_id,
                num_augmenters=self.num_augmenters
            )
            positive_graph = self.sql_to_graph_with_features.convert_to_pyg(augmented_nx_graph)
            positive_graphs.append(positive_graph)
        positive_time = time.time() - start_positive_time

        # Step 3: Generate negatives (if applicable)
        start_negative_time = time.time()
        if self.num_negatives > 0:
            negative_indices = self._select_negative_indices(idx)
            negative_graphs = [
                self.sql_to_graph_with_features.convert_to_pyg(self.records[neg_idx]["graph"])
                for neg_idx in negative_indices
            ]
        else:
            negative_graphs = []
        negative_time = time.time() - start_negative_time

        return anchor_graph, positive_graphs, negative_graphs

    def _select_negative_indices(self, anchor_idx):
        """
        Select negative indices, ensuring they are from different databases.
        :param anchor_idx: Index of the anchor.
        :return: List of negative indices.
        """
        anchor_db_id = self.records[anchor_idx]["db_id"]
        valid_indices = [
            i for i, record in enumerate(self.records) if record["db_id"] != anchor_db_id
        ]
        if len(valid_indices) < self.num_negatives:
            return valid_indices
        return random.sample(valid_indices, self.num_negatives)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to batch anchor, positive, and negative graphs.
        """
        anchors, positives, negatives = zip(*batch)

        # Batch anchors
        batched_anchors = Batch.from_data_list(anchors)

        # Ensure all positives have the same length
        max_positives = max(len(pos) for pos in positives)
        padded_positives = []
        for pos in positives:
            padded_pos = pos + [pos[-1]] * (max_positives - len(pos))  # Pad with the last positive
            padded_positives.append(padded_pos)
        batched_positives = [
            Batch.from_data_list([pos[i] for pos in padded_positives if len(pos) > i])
            for i in range(max_positives)
        ]

        # Ensure all negatives have the same length
        max_negatives = max(len(neg) for neg in negatives)
        padded_negatives = []
        for neg in negatives:
            padded_neg = neg + [neg[-1]] * (max_negatives - len(neg))  # Pad with the last negative
            padded_negatives.append(padded_neg)
        batched_negatives = [
            Batch.from_data_list([neg[i] for neg in padded_negatives if len(neg) > i])
            for i in range(max_negatives)
        ]

        return batched_anchors, batched_positives, batched_negatives
