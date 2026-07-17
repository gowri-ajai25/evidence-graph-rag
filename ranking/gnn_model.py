"""
Heterogeneous GraphSAGE model for evidence-aware treatment ranking.

Architecture:
  - Two-layer heterogeneous GraphSAGE encoder, producing embeddings for
    Patient, Gene, and Drug nodes by message-passing across all edge types.
  - A scoring head (dot product + MLP) that predicts the evidence-weighted
    relevance of a (Patient, Drug) pair given their learned embeddings,
    using the patient's mutated-gene -> drug evidence paths as supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()

        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr="mean")

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), out_channels)
            for edge_type in metadata[1]
        }, aggr="mean")

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class TreatmentRanker(nn.Module):
    """
    Full model: encodes the heterogeneous graph, then scores Gene-Drug pairs
    (which we use as a proxy for Patient-Drug relevance via mutated genes).
    """
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, out_channels, metadata)

        # MLP scoring head: takes concatenated [gene_emb, drug_emb] -> relevance score
        self.score_head = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def score_pairs(self, gene_emb, drug_emb, gene_indices, drug_indices):
        """Scores specific (gene, drug) pairs given their embeddings."""
        g = gene_emb[gene_indices]
        d = drug_emb[drug_indices]
        combined = torch.cat([g, d], dim=1)
        return self.score_head(combined).squeeze(-1)