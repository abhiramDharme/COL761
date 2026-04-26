"""
train.py – COL761 A3 GNN training script for datasets A, B, C.

Usage:
    python train.py --dataset A --data_dir /path/to/public_datasets \
        --model_dir ./models --kerberos cs5221139
    python train.py --dataset B --data_dir /path/to/public_datasets \
        --model_dir ./models --kerberos cs5221139
    python train.py --dataset C --data_dir /path/to/public_datasets \
        --model_dir ./models --kerberos cs5221139
"""

import argparse
import os
import random
import sys
import time

import numpy as np
# Use a conservative allocator split strategy to avoid large-fragment issues.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import RandomNodeLoader
from sklearn.metrics import roc_auc_score

# Add src to path so load_dataset works
sys.path.insert(0, os.path.dirname(__file__))
from load_dataset import load_dataset, COL761NodeDataset, COL761LinkDataset
from models import CoraGCN, NodeGNN, LinkGNN


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    # Enforce deterministic behavior as much as possible across machines.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False


def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    """Fraction of positives that rank in top-k against their hard negatives."""
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


def get_batch_node_ids(batch) -> torch.Tensor:
    """
    Returns global node IDs for a RandomNodeLoader batch.
    Handles PyG versions that expose either `n_id` or `node_id`.
    """
    if hasattr(batch, "n_id"):
        return batch.n_id
    if hasattr(batch, "node_id"):
        return batch.node_id
    raise AttributeError(
        "Batch is missing global node IDs (`n_id`/`node_id`). "
        "Populate `data.n_id = torch.arange(num_nodes)` before creating the loader."
    )


def sample_endpoint_corrupted_negatives(
    pos_edges: torch.Tensor,
    num_nodes: int,
    positive_edge_keys: torch.Tensor,
    max_resample: int = 5,
) -> torch.Tensor:
    """
    Build hard negatives by corrupting one endpoint of each positive edge.
    This mirrors dataset C evaluation, where each hard negative shares one
    endpoint with the corresponding positive edge.
    """
    device = pos_edges.device
    neg = pos_edges.clone()

    for _ in range(max_resample):
        bsz = neg.shape[0]
        flip = torch.rand(bsz, device=device) < 0.5
        rand_nodes = torch.randint(0, num_nodes, (bsz,), device=device)
        neg[flip, 0] = rand_nodes[flip]
        neg[~flip, 1] = rand_nodes[~flip]

        keys = neg[:, 0] * num_nodes + neg[:, 1]
        invalid = torch.isin(keys, positive_edge_keys) | (neg[:, 0] == neg[:, 1])
        if not invalid.any():
            return neg

        idx = torch.nonzero(invalid, as_tuple=False).view(-1)
        # Keep only invalid entries and resample in next loop.
        neg[idx] = pos_edges[idx]

    # Final pass: allow remaining collisions (rare) instead of an infinite loop.
    return neg


# ──────────────────────────────────────────────────────────────────────────────
# Dataset A – 7-class node classification
# ──────────────────────────────────────────────────────────────────────────────

def train_A(args, device):
    print("=== Training Dataset A (7-class node classification) ===")
    # Hardcoded best-known preset for Dataset A.
    active_seed = getattr(args, "effective_seed", args.seed)
    a_hidden_dim = 16
    a_dropout = 0.55
    a_lr = 1e-2
    a_weight_decay = 5e-4
    a_label_smoothing = 0.0
    a_normalize_input = True
    fixed_epochs = 2000
    fixed_patience = 220
    eval_every = 10

    ds = load_dataset("A", args.data_dir)
    data = ds[0].to(device)

    in_dim = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    print(f"  Nodes: {data.num_nodes:,}  Edges: {data.num_edges:,}")
    print(f"  Features: {in_dim}  Classes: {num_classes}")

    train_nodes = data.labeled_nodes[data.train_mask]
    val_nodes   = data.labeled_nodes[data.val_mask]
    train_labels = data.y[data.train_mask].long()
    val_labels   = data.y[data.val_mask].long()
    print(f"  Train: {len(train_nodes):,}  Val: {len(val_nodes):,}")

    model = CoraGCN(
        in_dim=in_dim,
        hidden_dim=a_hidden_dim,
        out_dim=num_classes,
        dropout=a_dropout,
        normalize_input=a_normalize_input,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=a_lr,
        weight_decay=a_weight_decay,
    )
    scheduler = None
    print(
        "  Model: CoraGCN [HARDCODED BEST] "
        f"(seed={active_seed}, hidden={a_hidden_dim}, dropout={a_dropout}, lr={a_lr}, "
        f"wd={a_weight_decay}, normalize_input={a_normalize_input})"
    )

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = 0.0
    best_state = None
    patience_count = 0

    for epoch in range(1, fixed_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[train_nodes],
            train_labels,
            label_smoothing=a_label_smoothing,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if epoch % eval_every == 0 or epoch == fixed_epochs:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                preds = logits.argmax(dim=1)
                val_acc = (preds[val_nodes] == val_labels).float().mean().item()
                train_acc = (preds[train_nodes] == train_labels).float().mean().item()

            print(f"  Ep {epoch:4d}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += eval_every
                if patience_count >= fixed_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"BEST_METRIC::{best_val_acc:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Dataset B – binary node classification (ROC-AUC)
# ──────────────────────────────────────────────────────────────────────────────

def train_B(args, device):
    print("=== Training Dataset B (binary node classification) ===")
    # Hardcoded best-known preset for Dataset B (cfg C baseline).
    active_seed = getattr(args, "effective_seed", args.seed)
    b_hidden_dim = 192
    b_num_layers = 3
    b_dropout = 0.35
    b_conv_type = "sage"
    b_jk_mode = "cat"
    b_use_residual = True
    b_classifier_mode = "mlp"
    b_lr = 1e-3
    b_weight_decay = 1e-4
    b_pos_weight_power = 0.75
    b_node_num_parts = 480
    b_eval_every = 10
    fixed_epochs = 220
    fixed_patience = 30
    b_time_budget_s = 7000

    ds = load_dataset("B", args.data_dir)
    data = ds[0]  # keep on CPU — 2.89M nodes × 1179 features won't fit on GPU

    in_dim = data.x.shape[1]
    N = data.num_nodes
    print(f"  Nodes: {N:,}  Edges: {data.num_edges:,}")
    print(f"  Features: {in_dim}")

    train_nodes = data.labeled_nodes[data.train_mask]
    val_nodes   = data.labeled_nodes[data.val_mask]
    train_labels = data.y[data.train_mask].float()
    val_labels   = data.y[data.val_mask].float()
    print(f"  Train: {len(train_nodes):,}  Val: {len(val_nodes):,}")

    # y_full maps global node ID → label for fast mini-batch label lookup
    y_full = torch.zeros(N, dtype=torch.float)
    y_full[train_nodes] = train_labels
    y_full[val_nodes]   = val_labels

    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    raw_pos_weight = n_neg / max(n_pos, 1)
    effective_pos_weight = raw_pos_weight ** b_pos_weight_power
    pos_weight = torch.tensor([effective_pos_weight], device=device)
    print(f"  Pos: {int(n_pos)}  Neg: {int(n_neg)}  pos_weight: {pos_weight.item():.2f}")

    # Boolean mask for fast per-batch training-node lookup
    train_mask_full = torch.zeros(N, dtype=torch.bool)
    train_mask_full[train_nodes] = True
    val_mask_full = torch.zeros(N, dtype=torch.bool)
    val_mask_full[val_nodes] = True

    # Make global node ids available to subgraphs across PyG versions.
    if not hasattr(data, "n_id"):
        data.n_id = torch.arange(N, dtype=torch.long)

    model = NodeGNN(
        in_dim=in_dim,
        hidden_dim=b_hidden_dim,
        out_dim=1,
        num_layers=b_num_layers,
        dropout=b_dropout,
        conv_type=b_conv_type,
        jk_mode=b_jk_mode,
        use_residual=b_use_residual,
        classifier_mode=b_classifier_mode,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        "  Config [HARDCODED BEST]: "
        f"seed={active_seed}, hidden={b_hidden_dim}, layers={b_num_layers}, dropout={b_dropout}, "
        f"conv={b_conv_type}, jk={b_jk_mode}, residual={b_use_residual}, "
        f"head={b_classifier_mode}, lr={b_lr}, wd={b_weight_decay}, "
        f"pos_weight_power={b_pos_weight_power}, node_num_parts={b_node_num_parts}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=b_lr, weight_decay=b_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=fixed_epochs, eta_min=1e-5)

    # RandomNodeLoader partitions the graph with torch subgraph() — no pyg-lib/torch-sparse needed
    train_loader = RandomNodeLoader(
        data, num_parts=b_node_num_parts, shuffle=True, num_workers=0
    )
    val_loader = RandomNodeLoader(
        data, num_parts=b_node_num_parts, shuffle=False, num_workers=0
    )
    eval_every = b_eval_every

    best_val_auc = 0.0
    best_state = None
    patience_count = 0
    train_t0 = time.time()
    deadline = train_t0 + b_time_budget_s

    for epoch in range(1, fixed_epochs + 1):
        if time.time() >= deadline:
            print(
                f"  Time budget reached before epoch {epoch} "
                f"(elapsed={time.time() - train_t0:.1f}s, budget={b_time_budget_s}s)."
            )
            break
        epoch_t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        stop_due_to_time = False
        for batch in train_loader:
            if time.time() >= deadline:
                stop_due_to_time = True
                break
            n_id = get_batch_node_ids(batch)
            is_train = train_mask_full[n_id]
            if not is_train.any():
                continue
            edge_index = batch.edge_index
            if edge_index.numel() == 0:
                continue
            num_batch_nodes = batch.x.size(0)
            edge_min = int(edge_index.min().item())
            edge_max = int(edge_index.max().item())
            if edge_min < 0 or edge_max >= num_batch_nodes:
                print(
                    f"    [WARN] Skipping invalid train batch "
                    f"(edge range [{edge_min}, {edge_max}], nodes={num_batch_nodes})."
                )
                continue
            try:
                optimizer.zero_grad()
                out = model(
                    batch.x.to(device),
                    edge_index.long().contiguous().to(device),
                ).squeeze(-1)
                out_train = out[is_train]
                labels = y_full[n_id[is_train]].to(device)
                loss = F.binary_cross_entropy_with_logits(out_train, labels, pos_weight=pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                msg = str(e)
                if "CUDA driver error: invalid argument" in msg:
                    print("    [CUDA] Skipping one invalid train batch.")
                    continue
                if "CUDACachingAllocator" in msg or "INTERNAL ASSERT FAILED" in msg:
                    print("    [CUDA-ALLOC] Skipping one unstable train batch.")
                    continue
                if "out of memory" in msg.lower():
                    print("    [OOM] Skipping one oversized train batch.")
                    continue
                raise
                continue
        if stop_due_to_time:
            print(
                f"  Time budget reached during epoch {epoch} "
                f"(elapsed={time.time() - train_t0:.1f}s, budget={b_time_budget_s}s)."
            )
            break
        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % args.b_train_log_every == 0:
            print(
                f"  Ep {epoch:4d}  train_loss={avg_loss:.4f}  "
                f"batches={n_batches:3d}  time={time.time() - epoch_t0:.1f}s",
                flush=True,
            )

        if epoch % eval_every == 0 or epoch == fixed_epochs:
            if time.time() >= deadline:
                print(
                    f"  Time budget reached before validation at epoch {epoch} "
                    f"(elapsed={time.time() - train_t0:.1f}s, budget={b_time_budget_s}s)."
                )
                break
            model.eval()
            val_scores = []
            val_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    if time.time() >= deadline:
                        stop_due_to_time = True
                        break
                    n_id = get_batch_node_ids(batch)
                    is_val = val_mask_full[n_id]
                    if not is_val.any():
                        continue
                    edge_index = batch.edge_index
                    if edge_index.numel() == 0:
                        continue
                    num_batch_nodes = batch.x.size(0)
                    edge_min = int(edge_index.min().item())
                    edge_max = int(edge_index.max().item())
                    if edge_min < 0 or edge_max >= num_batch_nodes:
                        print(
                            f"    [WARN] Skipping invalid val batch "
                            f"(edge range [{edge_min}, {edge_max}], nodes={num_batch_nodes})."
                        )
                        continue
                    try:
                        out = model(
                            batch.x.to(device),
                            edge_index.long().contiguous().to(device),
                        ).squeeze(-1)
                        val_scores.append(torch.sigmoid(out[is_val]).cpu())
                        val_targets.append(y_full[n_id[is_val]].cpu())
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                        msg = str(e)
                        if "CUDA driver error: invalid argument" in msg:
                            print("    [CUDA] Skipping one invalid val batch.")
                            continue
                        if "CUDACachingAllocator" in msg or "INTERNAL ASSERT FAILED" in msg:
                            print("    [CUDA-ALLOC] Skipping one unstable val batch.")
                            continue
                        if "out of memory" in msg.lower():
                            print("    [OOM] Skipping one oversized val batch.")
                            continue
                        raise
                        continue
            if stop_due_to_time:
                print(
                    f"  Time budget reached during validation at epoch {epoch} "
                    f"(elapsed={time.time() - train_t0:.1f}s, budget={b_time_budget_s}s)."
                )
                break

            if val_scores:
                val_auc = roc_auc_score(
                    torch.cat(val_targets).numpy(),
                    torch.cat(val_scores).numpy(),
                )
            else:
                val_auc = 0.5

            print(f"  Ep {epoch:4d}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += eval_every
                if patience_count >= fixed_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    print(f"  Best val AUC-ROC: {best_val_auc:.4f}")
    print(f"BEST_METRIC::{best_val_auc:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Dataset C – link prediction (Hits@50)
# ──────────────────────────────────────────────────────────────────────────────

def train_C(args, device):
    print("=== Training Dataset C (link prediction) ===")
    # Hardcoded best-known preset for Dataset C.
    active_seed = getattr(args, "effective_seed", args.seed)
    c_hidden_dim = 256
    c_num_layers = 3
    c_dropout = 0.20
    c_conv_type = "gcn"
    c_decoder_mode = "sym"
    c_lr = 2e-3
    c_weight_decay = 1e-4
    c_steps_per_epoch = 4
    c_edge_batch_size = 4096
    c_neg_pos_ratio = 1.5
    c_rank_weight = 0.8
    c_margin = 0.2
    c_eval_every = 10
    c_use_endpoint_negs = True
    c_smooth_window = 1
    fixed_epochs = 300
    fixed_patience = 80

    ds = load_dataset("C", args.data_dir)

    # Move tensors to device
    x          = ds.x.to(device) if hasattr(ds, "x") else None
    edge_index = ds.edge_index.to(device)
    train_pos  = ds.train_pos.to(device)   # [M, 2]
    train_neg  = ds.train_neg.to(device)   # [M, 2]  (may be [M, 2] or larger)
    valid_pos  = ds.valid_pos.to(device)   # [V, 2]
    valid_neg  = ds.valid_neg.to(device)   # [V, K, 2]

    if train_neg.dim() == 3:
        train_neg = train_neg.view(-1, 2)

    num_nodes = ds.num_nodes
    in_dim = x.shape[1] if x is not None else 64

    # If no features, use learned node embeddings
    node_emb = None
    if x is None:
        node_emb = torch.nn.Embedding(num_nodes, in_dim).to(device)
        x = node_emb.weight

    V, K, _ = valid_neg.shape
    print(f"  Nodes: {num_nodes:,}  Edges: {edge_index.shape[1]:,}")
    print(f"  Features: {in_dim}")
    print(f"  train_pos: {train_pos.shape}  valid_pos: {valid_pos.shape}")
    print(f"  valid_neg: {valid_neg.shape}")

    model = LinkGNN(
        in_dim=in_dim,
        hidden_dim=c_hidden_dim,
        num_layers=c_num_layers,
        dropout=c_dropout,
        conv_type=c_conv_type,
        decoder_mode=c_decoder_mode,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        "  Config [HARDCODED BEST]: "
        f"seed={active_seed}, hidden={c_hidden_dim}, layers={c_num_layers}, dropout={c_dropout}, "
        f"conv={c_conv_type}, lr={c_lr}, wd={c_weight_decay}, decoder={c_decoder_mode}, "
        f"neg_pos_ratio={c_neg_pos_ratio}, margin={c_margin}, "
        f"endpoint_negs={c_use_endpoint_negs}, smooth_window={c_smooth_window}"
    )

    params = list(model.parameters())
    if node_emb is not None:
        params += list(node_emb.parameters())
    optimizer = torch.optim.AdamW(params, lr=c_lr, weight_decay=c_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=fixed_epochs, eta_min=1e-5)

    # Pre-flatten valid_neg for scoring: [V*K, 2]
    valid_neg_flat = valid_neg.view(V * K, 2)

    best_val_hits = 0.0
    best_smooth_hits = 0.0
    best_state = None
    patience_count = 0
    val_hits_history = []

    # Mini-batch training over edge pairs
    pos_batch_size = min(c_edge_batch_size, len(train_pos))
    neg_batch_size = max(
        pos_batch_size,
        int(pos_batch_size * c_neg_pos_ratio),
    )
    eval_every = c_eval_every
    num_train_neg = len(train_neg)
    num_train_pos = len(train_pos)
    pos_keys = train_pos[:, 0] * num_nodes + train_pos[:, 1]
    rev_pos_keys = train_pos[:, 1] * num_nodes + train_pos[:, 0]
    positive_edge_keys = torch.unique(torch.cat([pos_keys, rev_pos_keys], dim=0))
    hard_neg_bsz = min(pos_batch_size, neg_batch_size)
    rand_neg_bsz = max(0, neg_batch_size - hard_neg_bsz)

    for epoch in range(1, fixed_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(c_steps_per_epoch):
            optimizer.zero_grad()

            pos_idx = torch.randint(num_train_pos, (pos_batch_size,), device=device)
            pos_batch = train_pos[pos_idx]   # [Bp, 2]
            if c_use_endpoint_negs:
                neg_hard = sample_endpoint_corrupted_negatives(
                    pos_batch[:hard_neg_bsz],
                    num_nodes=num_nodes,
                    positive_edge_keys=positive_edge_keys,
                )
            else:
                neg_idx = torch.randint(num_train_neg, (hard_neg_bsz,), device=device)
                neg_hard = train_neg[neg_idx]

            if rand_neg_bsz > 0:
                neg_idx = torch.randint(num_train_neg, (rand_neg_bsz,), device=device)
                neg_rand = train_neg[neg_idx]
                neg_batch = torch.cat([neg_hard, neg_rand], dim=0)
            else:
                neg_batch = neg_hard

            z = model.encode(x, edge_index)
            pos_scores = model.decode(z, pos_batch)
            neg_scores = model.decode(z, neg_batch)

            bce_pos = F.binary_cross_entropy_with_logits(
                pos_scores,
                torch.ones_like(pos_scores),
            )
            bce_neg = F.binary_cross_entropy_with_logits(
                neg_scores,
                torch.zeros_like(neg_scores),
            )
            bce_loss = bce_pos + bce_neg

            rank_neg_scores = model.decode(z, neg_hard)
            rank_pos_scores = pos_scores[:rank_neg_scores.shape[0]]
            rank_loss = F.relu(
                c_margin - rank_pos_scores + rank_neg_scores
            ).mean()

            loss = bce_loss + c_rank_weight * rank_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if epoch % eval_every == 0 or epoch == fixed_epochs:
            model.eval()
            with torch.no_grad():
                z = model.encode(x, edge_index)
                pos_scores = model.decode(z, valid_pos)               # [V]
                neg_scores = model.decode(z, valid_neg_flat).view(V, K)  # [V, K]
                val_hits = hits_at_k(pos_scores, neg_scores, k=50)

            val_hits_history.append(val_hits)
            smooth_hits = float(np.mean(val_hits_history[-c_smooth_window:]))
            avg_loss = epoch_loss / max(c_steps_per_epoch, 1)
            print(
                f"  Ep {epoch:4d}  loss={avg_loss:.4f}  "
                f"val_hits@50={val_hits:.4f}  smooth={smooth_hits:.4f}"
            )

            if smooth_hits > best_smooth_hits:
                best_smooth_hits = smooth_hits
                best_val_hits = val_hits
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += eval_every
                if patience_count >= fixed_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    print(f"  Best val Hits@50: {best_val_hits:.4f}")
    print(f"BEST_METRIC::{best_val_hits:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().cpu()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COL761 A3 GNN trainer")
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       default=None, choices=["node", "link"],
                        help="Optional task type for evaluator compatibility")
    parser.add_argument("--data_dir",   required=True,
                        help="Absolute path to public_datasets directory")
    parser.add_argument("--model_dir",  default="./models")
    parser.add_argument("--kerberos",   default="cs5221139")
    parser.add_argument("--epochs",     type=int, default=1000)
    parser.add_argument("--patience",   type=int, default=150,
                        help="Early-stopping patience (epochs)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for node GNN (A and B)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of GNN layers (A and B)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate for node GNN (A and B)")
    parser.add_argument("--conv_type", default="sage", choices=["sage", "gcn", "gat"],
                        help="GNN layer type for node GNN (A and B)")
    parser.add_argument("--jk_mode", default="cat", choices=["last", "cat", "max"],
                        help="Jumping Knowledge aggregation for node GNN (A and B)")
    parser.add_argument("--use_residual", type=int, default=1, choices=[0, 1],
                        help="Use residual skip connections in node GNN (A and B)")
    parser.add_argument("--classifier_mode", default="mlp", choices=["mlp", "linear"],
                        help="Classifier head type for node GNN (A and B)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for node GNN (A and B)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for node GNN (A and B)")
    parser.add_argument("--pos_weight_power", type=float, default=1.0,
                        help="Use pos_weight^power for dataset B BCE loss")
    parser.add_argument("--node_num_parts", type=int, default=120,
                        help="Number of random node partitions for Dataset B")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate every N epochs")
    parser.add_argument("--b_train_log_every", type=int, default=1,
                        help="Print training-progress line every N epochs for dataset B")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--link_batch_size", type=int, default=65536,
                        help="Mini-batch size for link-prediction training")
    parser.add_argument("--c_hidden_dim", type=int, default=256,
                        help="Hidden dimension for dataset C LinkGNN")
    parser.add_argument("--c_num_layers", type=int, default=3,
                        help="Number of GNN layers for dataset C")
    parser.add_argument("--c_dropout", type=float, default=0.25,
                        help="Dropout for dataset C")
    parser.add_argument("--c_conv_type", default="sage", choices=["sage", "gcn"],
                        help="Convolution type for dataset C")
    parser.add_argument("--c_decoder_mode", default="sym", choices=["sym", "concat"],
                        help="Edge decoder mode for dataset C")
    parser.add_argument("--c_lr", type=float, default=2e-3,
                        help="Learning rate for dataset C")
    parser.add_argument("--c_weight_decay", type=float, default=1e-4,
                        help="Weight decay for dataset C")
    parser.add_argument("--c_steps_per_epoch", type=int, default=4,
                        help="Number of optimization steps per epoch for dataset C")
    parser.add_argument("--c_edge_batch_size", type=int, default=None,
                        help="Positive edge batch size for dataset C")
    parser.add_argument("--c_neg_pos_ratio", type=float, default=1.5,
                        help="Negative:positive sampling ratio for dataset C")
    parser.add_argument("--c_rank_weight", type=float, default=0.5,
                        help="Weight of ranking loss term for dataset C")
    parser.add_argument("--c_margin", type=float, default=0.2,
                        help="Margin for ranking loss in dataset C")
    parser.add_argument("--c_eval_every", type=int, default=10,
                        help="Evaluate every N epochs for dataset C")
    parser.add_argument("--c_use_endpoint_negs", type=int, default=1, choices=[0, 1],
                        help="Use endpoint-corrupted hard negatives for dataset C")
    parser.add_argument("--a_model", default="cora_gcn", choices=["cora_gcn", "nodegnn"],
                        help="Architecture for dataset A")
    parser.add_argument("--a_hidden_dim", type=int, default=16,
                        help="Hidden dimension for CoraGCN on dataset A")
    parser.add_argument("--a_dropout", type=float, default=0.5,
                        help="Dropout for CoraGCN on dataset A")
    parser.add_argument("--a_lr", type=float, default=1e-2,
                        help="Learning rate for CoraGCN on dataset A")
    parser.add_argument("--a_weight_decay", type=float, default=5e-4,
                        help="Weight decay for CoraGCN on dataset A")
    parser.add_argument("--a_label_smoothing", "--label_smoothing", dest="a_label_smoothing",
                        type=float, default=0.0,
                        help="Label smoothing for dataset A")
    parser.add_argument("--a_normalize_input", type=int, default=1, choices=[0, 1],
                        help="Enable row-normalization of input features for dataset A")
    args = parser.parse_args()

    if args.task is not None:
        valid = {"node": ("A", "B"), "link": ("C",)}
        if args.dataset not in valid[args.task]:
            parser.error(
                f"--task {args.task} is not valid for --dataset {args.dataset}. "
                f"Expected dataset in {valid[args.task]}."
            )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(args.data_dir)
    if args.c_edge_batch_size is None:
        args.c_edge_batch_size = args.link_batch_size

    hardcoded_dataset_seeds = {"A": 72, "B": 52, "C": 72}
    effective_seed = hardcoded_dataset_seeds.get(args.dataset, args.seed)
    args.effective_seed = effective_seed
    set_seed(args.effective_seed)
    os.makedirs(args.model_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    if args.dataset in hardcoded_dataset_seeds:
        print(f"Using seed: {args.effective_seed} [HARDCODED for dataset {args.dataset}]")
    else:
        print(f"Using seed: {args.effective_seed}")

    t0 = time.time()

    if args.dataset == "A":
        model = train_A(args, device)
    elif args.dataset == "B":
        model = train_B(args, device)
    else:
        model = train_C(args, device)

    model_path = os.path.join(args.model_dir,
                              f"{args.kerberos}_model_{args.dataset}.pt")
    torch.save(model, model_path)
    print(f"\nModel saved to: {model_path}  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
