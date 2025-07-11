import os
import random
from pathlib import Path
import json
import pickle

import torch
import numpy as np
from typing import List
from torch.nn.functional import cosine_similarity
from demonstration_selector.base_demonstration_selector import BaseDemonstrationSelector
from dataset_classes.base_dataset import BaseDataset
from gnn_contrastive_learning.model import GNNEncoderGCN, GNNEncoderGAT
from torch_geometric.data import Batch
from gnn_contrastive_learning.sql_to_graph import SQL2GraphWithFeatures, SQL2Graph

def load_model(checkpoint_path, encoder_type="gcn", input_dim=None, hidden_dim=64, output_dim=64, num_layers=2, readout="concat", dropout=0.5):
    """
    Load the GNN encoder model from a checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    ## search model_config.json in the same directory as the checkpoint
    if Path.exists(Path(checkpoint_path).parent / "model_config.json"):
        with open(Path(checkpoint_path).parent / "model_config.json", "r") as f:
            model_config = json.load(f)
        encoder_type = model_config.get("encoder_type", encoder_type)
        input_dim = model_config.get("input_dim", input_dim)
        hidden_dim = model_config.get("hidden_dim", hidden_dim)
        output_dim = model_config.get("output_dim", output_dim)
        num_layers = model_config.get("num_layers", num_layers)
        readout = model_config.get("readout", readout)
        dropout = model_config.get("dropout", dropout)

    if encoder_type == "gcn":
        encoder = GNNEncoderGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            readout=readout,
            dropout=dropout,
        )
    elif encoder_type == "gat":
        encoder = GNNEncoderGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            readout=readout,
            dropout=dropout,
            heads=4,
        )
    else:
        # raise ValueError(f"Invalid encoder type: {encoder_type}")
        encoder = GNNEncoder(
            input_dim=773,
            hidden_dim=128,
            output_dim=64,
            num_layers=2,
            readout="mean",
            dropout=0.5,
        )
    # encoder = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    return encoder


class GCLDemonstrationSelector(BaseDemonstrationSelector):
    """
    Generate demonstrations by selecting top-k instances based on cosine similarity
    using embeddings from a GCL-trained encoder.
    """
    def __init__(self, dataset: BaseDataset, encoder_path, sql_to_graph_with_features=None, device=None, cache_file_path=None):
        """
        :param dataset: The dataset containing demonstrations.
        :param encoder: The trained GCL encoder for computing embeddings.
        :param sql_to_graph_with_features: Utility for converting SQL queries to PyG graphs.
        :param device: Device to perform computations on ("cpu" or "cuda").
        """
        super().__init__(dataset)
        self.name = "gcl_demonstration_selector"
        if sql_to_graph_with_features is None:
            sql_to_graph_with_features = SQL2GraphWithFeatures(batch_size=32, model_name="all-mpnet-base-v2")
        self.sql_to_graph_with_features = sql_to_graph_with_features

        ## hard code the input_dim, hidden_dim, output_dim, num_layers, readout, dropout for models that don't have model_config.json
        llm_embedding_dim = sql_to_graph_with_features.model.get_sentence_embedding_dimension()
        node_type_encoding_dim = len(sql_to_graph_with_features.node_types)
        input_dim = llm_embedding_dim + node_type_encoding_dim
        hidden_dim = 128
        output_dim = 64
        num_layers = 2
        readout = "mean"
        dropout = 0.5
        encoder_type = "others"

        ## load the model
        self.encoder = load_model(encoder_path, encoder_type=encoder_type, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=readout, dropout=dropout)

        # check if cuda is available
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.encoder.to(self.device)
        self.encoder.eval()
        self.embeddings = self._precompute_embeddings(cache_file_path)

    def _precompute_embeddings(self, cache_file_path=None, flag_save_cache_if_not_exist=True):
        """
        Precompute embeddings for all demonstrations in the dataset.
        :return: embeddings
        """
        if cache_file_path is not None and os.path.exists(cache_file_path):
            print("Loading precomputed embeddings for demonstrations...")
            with open(cache_file_path, "rb") as f:
                embeddings = pickle.load(f)
            return embeddings
        embeddings = []
        print("Precomputing embeddings for demonstrations...")
        with torch.no_grad():
            for record in self.dataset.data["train"]:
                sql_query = record["query"]
                pyg_graph = self.sql_to_graph_with_features.sql_to_pyg(sql_query, flag_replace_double_quotes=True).to(self.device)
                embedding = self.encoder(pyg_graph.x, pyg_graph.edge_index, pyg_graph.batch)
                embeddings.append(embedding.cpu().numpy())
        embeddings = np.vstack(embeddings)  # Combine all embeddings into a single numpy array
        if flag_save_cache_if_not_exist and cache_file_path is not None:
            with open(cache_file_path, "wb") as f:
                pickle.dump(embeddings, f)
            print(f"Saved precomputed embeddings to {cache_file_path}")
        return embeddings

    def select_demonstrations(self, record_data: dict, num_demonstrations: int = 5, flag_return_ids: bool = False):
        """
        Select the top-k demonstrations based on cosine similarity.
        :param record_data: The record to use as the anchor for selecting demonstrations.
        :param num_demonstrations: Number of demonstrations to select.
        :param flag_return_ids: If True, return IDs instead of full records.
        :return: Selected demonstrations or their IDs.
        """
        # Compute embedding for the input record
        with torch.no_grad():
            sql_query = record_data["query"]
            pyg_graph = self.sql_to_graph_with_features.sql_to_pyg(sql_query, flag_replace_double_quotes=True).to(self.device)
            record_embedding = self.encoder(pyg_graph.x, pyg_graph.edge_index, pyg_graph.batch).cpu().numpy()

        # Compute cosine similarities
        similarities = np.dot(self.embeddings, record_embedding.T).flatten()

        # Select top-k demonstrations
        top_indices = np.argsort(similarities)[::-1][:num_demonstrations]
        selected_demonstrations = [self.demonstrations[idx] for idx in top_indices]

        if flag_return_ids:
            return [demo["idx"] for demo in selected_demonstrations]
        else:
            return selected_demonstrations

    def get_default_output_file_path(self, config: dict):
        """
        Get the default output file path to store the prompts.
        :param config: Configuration dictionary.
        :return: Path to the default output file.
        """
        return os.path.join(
            config["dataset_dir_path"],
            "prompts",
            f"{config['dataset_name']}_{config['split_name']}_gcl_num_demo_{config['num_demonstrations']}_{config['template_option']}.json"
        )
