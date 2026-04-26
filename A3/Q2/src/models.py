"""
models.py – GNN architectures for COL761 A3 node classification and link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, BatchNorm, JumpingKnowledge


# ──────────────────────────────────────────────────────────────────────────────
# Node classification GNN (Datasets A and B)
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """One GNN layer with BN + ReLU + optional skip connection."""

    def __init__(
        self,
        in_dim,
        out_dim,
        conv_type="sage",
        heads=4,
        dropout=0.3,
        use_residual=True,
    ):
        super().__init__()
        if conv_type == "gat":
            # GAT with multi-head, then project back to out_dim
            self.conv = GATConv(in_dim, out_dim // heads, heads=heads,
                                dropout=dropout, concat=True)
        elif conv_type == "sage":
            self.conv = SAGEConv(in_dim, out_dim)
        else:
            self.conv = GCNConv(in_dim, out_dim)

        self.norm = BatchNorm(out_dim)
        self.use_residual = use_residual
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.dropout = dropout

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        if self.use_residual:
            out = out + self.skip(x)
        return out


class NodeGNN(nn.Module):
    """
    Deep GNN for node classification.
    Supports GAT and SAGE convolutions, residual connections, JK aggregation.

    Interface: forward(x, edge_index) → logits [N, num_classes]
    (matches predict.py's expected signature)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 4,
        dropout: float = 0.3,
        conv_type: str = "sage",
        jk_mode: str = "last",   # "last" | "cat" | "max"
        heads: int = 4,
        use_residual: bool = True,
        classifier_mode: str = "mlp",  # "mlp" | "linear"
    ):
        super().__init__()
        self.dropout = dropout
        self.jk_mode = jk_mode

        self.blocks = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)

        for i in range(num_layers - 1):
            self.blocks.append(
                ResidualBlock(dims[i], hidden_dim, conv_type=conv_type,
                              heads=heads, dropout=dropout, use_residual=use_residual)
            )

        # Determine input dimension for the classifier
        if jk_mode == "cat":
            clf_in = hidden_dim * (num_layers - 1)
            self.jk = JumpingKnowledge(mode="cat")
        elif jk_mode == "max":
            clf_in = hidden_dim
            self.jk = JumpingKnowledge(mode="max")
        else:
            clf_in = hidden_dim
            self.jk = None

        if classifier_mode == "linear":
            self.classifier = nn.Linear(clf_in, out_dim)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )

    def forward(self, x, edge_index):
        xs = []
        for block in self.blocks:
            x = block(x, edge_index)
            xs.append(x)

        if self.jk is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]

        return self.classifier(x)


# Canonical 2-layer GCN for Cora-like transductive node classification.
class CoraGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        normalize_input: bool = True,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, cached=True, normalize=True)
        self.dropout = dropout
        self.normalize_input = normalize_input

    def forward(self, x, edge_index):
        if self.normalize_input:
            # Row-normalize sparse bag-of-words style features (Cora-like setup).
            denom = x.sum(dim=1, keepdim=True).clamp_min(1.0)
            x = x / denom
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Link prediction GNN (Dataset C)
# ──────────────────────────────────────────────────────────────────────────────

class LinkGNN(nn.Module):
    """
    GNN encoder + dot-product decoder for link prediction.

    Interface: forward(x, edge_index, edge_pairs) → scores [E]
      edge_pairs: LongTensor [E, 2] of (src, dst) node indices
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = "sage",
        decoder_mode: str = "sym",  # "sym" | "concat"
    ):
        super().__init__()
        self.dropout = dropout
        self.decoder_mode = decoder_mode

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        for _ in range(num_layers):
            if conv_type == "sage":
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))

        # MLP scorer on pair embeddings.
        # sym mode uses [src * dst, |src - dst|], which is permutation-invariant.
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x, edge_index):
        x = self.input_proj(x)
        x = F.relu(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return x

    def decode(self, z, edge_pairs):
        """Scores a batch of (src, dst) edge pairs."""
        src = z[edge_pairs[:, 0]]
        dst = z[edge_pairs[:, 1]]
        if self.decoder_mode == "concat":
            pair = torch.cat([src, dst], dim=-1)
        else:
            pair = torch.cat([src * dst, torch.abs(src - dst)], dim=-1)
        return self.scorer(pair).squeeze(-1)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)
