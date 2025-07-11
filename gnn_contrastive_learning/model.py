import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GATConv


class GNNEncoderGCN(nn.Module):
    """
    GNN Encoder with options for readout: mean, max, sum, or concatenated pooling.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, readout="mean", dropout=0.5):
        super(GNNEncoderGCN, self).__init__()
        self.num_layers = num_layers
        self.readout = readout.lower()
        self.dropout = nn.Dropout(dropout)

        # Validate readout type
        valid_readouts = {"mean", "max", "sum", "concat"}
        if self.readout not in valid_readouts:
            raise ValueError(f"Unsupported readout type: {self.readout}. Choose from {valid_readouts}.")

        # GCN Layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))  # Final layer with hidden_dim

        # Projection head
        if self.readout == "concat":
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x, edge_index, batch):
        # GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)

        # Readout
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "concat":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)

        # Projection head
        x = self.projection_head(x)
        return x


class GNNEncoderGAT(nn.Module):
    """
    GNN Encoder with GAT layers and options for readout: mean, max, sum, or concatenated pooling.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, readout="mean", dropout=0.5, heads=4):
        super(GNNEncoderGAT, self).__init__()
        self.num_layers = num_layers
        self.readout = readout.lower()
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

        # Validate readout type
        valid_readouts = {"mean", "max", "sum", "concat"}
        if self.readout not in valid_readouts:
            raise ValueError(f"Unsupported readout type: {self.readout}. Choose from {valid_readouts}.")

        # GAT Layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))  # Final layer

        # Determine projection head input size based on readout
        if self.readout == "concat":
            projection_input_dim = hidden_dim * heads * 3
        else:
            projection_input_dim = hidden_dim * heads

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(projection_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch):
        # GAT layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)

        # Readout
        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "concat":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)

        # Projection head
        x = self.projection_head(x)
        return x

def contrastive_loss(anchor_emb, positive_embs, negative_embs, temperature=0.5):
    """
    Compute contrastive loss for anchor-positive-negative triplets.
    """
    # Combine all positives into a single tensor
    pos_emb = torch.stack(positive_embs, dim=0).permute(1, 0, 2)  # (N, num_positives, D)
    pos_scores = torch.exp(F.cosine_similarity(anchor_emb.unsqueeze(1), pos_emb, dim=2) / temperature).sum(dim=1)

    # Combine all negatives into a single tensor
    neg_emb = (
        torch.cat([neg.unsqueeze(0) for neg in negative_embs], dim=0).permute(1, 0, 2)
        if negative_embs
        else torch.empty((anchor_emb.size(0), 0, anchor_emb.size(1)), device=anchor_emb.device)
    )
    neg_scores = torch.exp(F.cosine_similarity(anchor_emb.unsqueeze(1), neg_emb, dim=2) / temperature).sum(dim=1)

    # Compute loss
    loss = -torch.log(pos_scores / (pos_scores + neg_scores)).mean()
    return loss, pos_scores.mean().item(), neg_scores.mean().item()




def per_anchor_contrastive_loss_with_metrics(anchors, positives, negatives, temperature=0.5):
    """
    Compute contrastive loss for each anchor independently and calculate avg_pos_score and avg_neg_score.
    :param anchors: Tensor of anchor embeddings (N x D).
    :param positives: List of tensors of positive embeddings (N x D each).
    :param negatives: List of tensors of negative embeddings (N x D each).
    :param temperature: Temperature scaling factor for contrastive loss.
    :return: Per-anchor contrastive loss (averaged across the batch), avg_pos_score, avg_neg_score.
    """
    batch_size = anchors.size(0)
    total_loss = 0
    total_pos_score = 0
    total_neg_score = 0
    num_positives = 0
    num_negatives = 0

    for i in range(batch_size):
        # Select the i-th anchor
        anchor = anchors[i].unsqueeze(0)  # Shape: (1 x D)

        # Select its positives and negatives
        positive = positives[i]  # List of positive embeddings (num_positives x D)
        negative = negatives[i] if negatives else []  # List of negative embeddings (num_negatives x D)

        # Calculate similarity scores
        pos_scores = torch.exp(F.cosine_similarity(anchor, positive, dim=1) / temperature)  # Shape: (num_positives)
        neg_scores = torch.exp(F.cosine_similarity(anchor, negative, dim=1) / temperature)  # Shape: (num_negatives)

        # Update positive and negative score totals
        total_pos_score += pos_scores.sum().item()
        total_neg_score += neg_scores.sum().item()
        num_positives += len(positive)
        num_negatives += len(negative)

        # Compute denominator (sum of positives and negatives)
        total_scores = pos_scores.sum() + neg_scores.sum()

        # Compute contrastive loss for the anchor
        anchor_loss = -torch.log(pos_scores.sum() / total_scores)
        total_loss += anchor_loss

    # Compute averages
    avg_pos_score = total_pos_score / num_positives if num_positives > 0 else 0
    avg_neg_score = total_neg_score / num_negatives if num_negatives > 0 else 0

    # Average the loss across all anchors
    avg_loss = total_loss / batch_size
    return avg_loss, avg_pos_score, avg_neg_score

