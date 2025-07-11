import time
import os
from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np

from model import GNNEncoderGCN, GNNEncoderGAT, contrastive_loss, per_anchor_contrastive_loss_with_metrics
from gnn_contrastive_learning.sql_to_graph import SQL2GraphWithFeatures
from gnn_contrastive_learning.sql_graph_augmenter import SQLGraphAugmenter
from gnn_contrastive_learning.gcl_dataset import GCLDataset
from dataset_classes.spider_dataset import SpiderDataset

def save_model(model, optimizer, epoch, model_path="./checkpoints", model_config:dict=None):
    """
    Save the model state, optimizer state, and epoch number.
    :param model: The model to save.
    :param optimizer: The optimizer to save.
    :param epoch: The current epoch.
    :param save_dir: Directory to save the checkpoints.
    """
    Path(model_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_path, f"model_epoch_{epoch}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }, model_path)
    if model_config is not None:
        with open(os.path.join(model_path, "model_config.json"), "w") as f:
            json.dump(model_config, f)
    print(f"Model saved to {model_path}")


def train_gcl(dataset, encoder, learning_rate=1e-3, weight_decay=1e-4, epochs=10, batch_size=64, temperature=0.5, save_interval=5, model_path="checkpoints", model_config:dict=None):
    """
    Train the GNN contrastive learning model with metrics monitoring.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    encoder = encoder.to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2 regularization
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    encoder.train()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        total_pos_score = 0
        total_neg_score = 0
        count = 0
        all_cos_sim = []  # Store cosine similarity statistics

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        # for batch in dataloader:
            batch_start_time = time.time()
            optimizer.zero_grad()
            
            # Unpack batch
            anchors, positives, negatives = batch
            
            # Move anchors to device
            anchors = anchors.to(device)
            positives = [pos.to(device) for pos in positives]
            negatives = [neg.to(device) for neg in negatives]
            
            # Encode anchor embeddings
            anchor_emb = encoder(anchors.x, anchors.edge_index, anchors.batch)  # Shape: (batch_size, D)
            
            # Process positives and negatives per anchor
            losses = []
            avg_pos_scores = []
            avg_neg_scores = []
            
            for i in range(len(anchor_emb)):
                # Encode the positive graphs
                positive_embs = [encoder(pos.x, pos.edge_index, pos.batch) for pos in positives]

                # Encode the negative graphs (if applicable)
                negative_embs = [encoder(neg.x, neg.edge_index, neg.batch) for neg in negatives]
                    
                # Compute loss for the anchor
                loss, avg_pos_score, avg_neg_score = per_anchor_contrastive_loss_with_metrics(
                    anchor_emb[i].unsqueeze(0),  # Anchor embedding
                    positive_embs,
                    negative_embs,
                    temperature
                )
                losses.append(loss)
                avg_pos_scores.append(avg_pos_score)
                avg_neg_scores.append(avg_neg_score)
            
            # Aggregate losses
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            total_pos_score += avg_pos_score
            total_neg_score += avg_neg_score
            count += 1

            # Compute alignment/uniformity metrics
            cos_sim = F.cosine_similarity(anchor_emb.unsqueeze(1), anchor_emb.unsqueeze(0), dim=2)
            all_cos_sim.extend(cos_sim.cpu().detach().numpy().flatten())
            
            # show batch time and loss
            # print(f"Batch Loss: {loss.item():.4f}. Batch Time: {time.time() - batch_start_time:.4f}s")

        # Epoch Metrics
        avg_loss = epoch_loss / count
        avg_pos_score = total_pos_score / count
        avg_neg_score = total_neg_score / count
        cos_sim_mean = np.mean(all_cos_sim)
        cos_sim_std = np.std(all_cos_sim)

        print(
            f"Epoch Time: {time.time() - epoch_start_time:.4f}s\n"
            f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}, "
            f"Avg Positive Score: {avg_pos_score:.4f}, Avg Negative Score: {avg_neg_score:.4f}, "
            f"Cosine Similarity: Mean={cos_sim_mean:.4f}, Std={cos_sim_std:.4f}\n"
            f"=================================================================="
        )

        # Save model at regular intervals
        if (epoch + 1) % save_interval == 0 or epoch + 1 == epochs:
            checkpoint_path = os.path.join(model_path, f"checkpoints")
            save_model(encoder, optimizer, epoch + 1, checkpoint_path, model_config)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN contrastive learning model")
    parser.add_argument("--dataset_path", type=str, default="datasets/spider", help="Path to the dataset")
    parser.add_argument("--model_path", type=str, default="gnn_contrastive_learning/models/v2", help="Path to save models")
    parser.add_argument("--encoder_type", type=str, choices=["gcn", "gat"], default="gcn", help="Type of GNN encoder")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--output_dim", type=int, default=64, help="Output dimension size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--readout", type=str, choices=["mean", "max", "sum", "concat"], default="concat", help="Readout function")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for contrastive loss")
    parser.add_argument("--save_interval", type=int, default=5, help="Save interval for checkpoints")
    parser.add_argument("--prob_feature_masking", type=float, default=1.0, help="Probability for feature masking")
    parser.add_argument("--prob_value_replacement", type=float, default=1.0, help="Probability for value replacement")
    parser.add_argument("--prob_keyword_replacement", type=float, default=1.0, help="Probability for keyword replacement")
    parser.add_argument("--prob_predicate_modification", type=float, default=1.0, help="Probability for predicate modification")
    parser.add_argument("--prob_join_node_removal", type=float, default=1.0, help="Probability for join node removal")
    parser.add_argument("--prob_table_column_replacement", type=float, default=1.0, help="Probability for table column replacement")
    parser.add_argument("--num_augmenters", type=int, default=1, help="Number of SQL graph augmenters")
    args = parser.parse_args()

    dataset = SpiderDataset(args.dataset_path)
    records = dataset.data["train"]
    sql_to_graph_with_features = SQL2GraphWithFeatures(
        batch_size=32, 
        model_name="all-mpnet-base-v2"
    )
    operator_probabilities = {
        "feature_masking": args.prob_feature_masking,
        "value_replacement": args.prob_value_replacement,
        "keyword_replacement": args.prob_keyword_replacement,
        "predicate_modification": args.prob_predicate_modification,
        "join_node_removal": args.prob_join_node_removal,
        "table_column_replacement": args.prob_table_column_replacement,
    }
    sql_graph_augmenter = SQLGraphAugmenter(
        src_schema_file=dataset.train_table_schema_path, 
        aug_schema_file=dataset.dev_table_schema_path,
        operator_probabilities=operator_probabilities,
    )
    dataset = GCLDataset(
        records, 
        sql_to_graph_with_features, 
        sql_graph_augmenter, 
        num_positives=3, 
        num_negatives=5,
        num_augmenters=args.num_augmenters
    )
    print(f"Dataset size: {len(records)}")

    llm_embedding_dim = sql_to_graph_with_features.model.get_sentence_embedding_dimension()
    node_type_encoding_dim = len(sql_to_graph_with_features.node_types)
    input_dim = llm_embedding_dim + node_type_encoding_dim

    if args.encoder_type == "gcn":
        encoder = GNNEncoderGCN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            readout=args.readout,
            dropout=args.dropout,
        )
    elif args.encoder_type == "gat":
        encoder = GNNEncoderGAT(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            readout=args.readout,
            dropout=args.dropout,
            heads=4,  # Default number of heads for GAT
        )
    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}")

    model_config = {
        "encoder_type": args.encoder_type,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "num_layers": args.num_layers,
        "readout": args.readout,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "save_interval": args.save_interval,
        "model_path": args.model_path,
        "prob_feature_masking": args.prob_feature_masking,
        "prob_value_replacement": args.prob_value_replacement,
        "prob_keyword_replacement": args.prob_keyword_replacement,
        "prob_predicate_modification": args.prob_predicate_modification,
        "prob_join_node_removal": args.prob_join_node_removal,
        "prob_table_column_replacement": args.prob_table_column_replacement,
    }
    train_gcl(
        dataset=dataset,
        encoder=encoder,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        save_interval=args.save_interval,
        model_path=args.model_path,
        model_config=model_config
    )