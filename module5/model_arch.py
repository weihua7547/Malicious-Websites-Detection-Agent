import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, global_mean_pool
from torch_geometric.data import Batch as PyGBatch


# =========================================================
# TAGCN Encoder
# =========================================================
class TagcnEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.convs.append(TAGConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, x, edge_index):
        if x.std(dim=0).mean() > 0:
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        else:
            x = torch.zeros_like(x)
        x = torch.nan_to_num(x, nan=0.0)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# =========================================================
# RAC Graph Classifier
# =========================================================
class RACGraphClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.pool = global_mean_pool
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def encode_graph(self, batch):
        z = self.encoder(batch.x, batch.edge_index)
        g = self.pool(z, batch.batch)
        g = torch.clamp(g, min=-10.0, max=10.0)
        g = torch.nan_to_num(g, nan=0.0)
        return g

    def forward(self, batch_samples, device):
        labels = torch.stack([b["labels"] for b in batch_samples]).to(device)
        q_graphs = [b["q_graph"] for b in batch_samples]
        q_batch = PyGBatch.from_data_list(q_graphs).to(device)
        q_emb = self.encode_graph(q_batch)

        max_neighbors = max(len(b["neighbors"]) for b in batch_samples)
        neighbor_graphs = []
        attn_mask = torch.zeros(len(batch_samples), max_neighbors, dtype=torch.bool, device=device)
        for i, b in enumerate(batch_samples):
            ng = b["neighbors"]
            num_ng = len(ng)
            attn_mask[i, :num_ng] = True
            if num_ng < max_neighbors:
                ng += [ng[-1] if ng else b["q_graph"]] * (max_neighbors - num_ng)
            neighbor_graphs.extend(ng)

        n_batch = PyGBatch.from_data_list(neighbor_graphs).to(device)
        n_emb = self.encode_graph(n_batch)
        n_emb = n_emb.view(len(batch_samples), max_neighbors, -1)

        key_padding_mask = ~attn_mask
        attn_out, _ = self.attn(
            q_emb.unsqueeze(1),
            n_emb,
            n_emb,
            key_padding_mask=key_padding_mask
        )
        attn_out = attn_out.squeeze(1)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)

        fused = torch.cat([q_emb, attn_out], dim=1)
        logits = self.classifier(fused)
        return logits, labels

    def get_embedding(self, batch_samples, device):
        labels = torch.stack([b["labels"] for b in batch_samples]).to(device)
        q_graphs = [b["q_graph"] for b in batch_samples]
        q_batch = PyGBatch.from_data_list(q_graphs).to(device)
        q_emb = self.encode_graph(q_batch)

        max_neighbors = max(len(b["neighbors"]) for b in batch_samples)
        neighbor_graphs = []
        attn_mask = torch.zeros(len(batch_samples), max_neighbors, dtype=torch.bool, device=device)
        for i, b in enumerate(batch_samples):
            ng = b["neighbors"]
            num_ng = len(ng)
            attn_mask[i, :num_ng] = True
            if num_ng < max_neighbors:
                ng += [ng[-1] if ng else b["q_graph"]] * (max_neighbors - num_ng)
            neighbor_graphs.extend(ng)

        n_batch = PyGBatch.from_data_list(neighbor_graphs).to(device)
        n_emb = self.encode_graph(n_batch)
        n_emb = n_emb.view(len(batch_samples), max_neighbors, -1)

        key_padding_mask = ~attn_mask
        attn_out, _ = self.attn(
            q_emb.unsqueeze(1),
            n_emb,
            n_emb,
            key_padding_mask=key_padding_mask
        )
        attn_out = attn_out.squeeze(1)
        attn_out = torch.nan_to_num(attn_out, nan=0.0)

        fused = torch.cat([q_emb, attn_out], dim=1)
        fused = fused.view(fused.size(0), -1)
        return fused
# =========================================================
# 基礎 MLP Fusion Module
# =========================================================
class TriModalMLPFusion(nn.Module):
    def __init__(self, modalities, bert_dim=768, rac_dim=128, 
                 num_trees=244, num_leaves=31, leaf_emb_dim=16,
                 mlp_hidden_dim=512, num_classes=4):
        super(TriModalMLPFusion, self).__init__()
        self.modalities = modalities
        
        # 1. BERT Projection + Heavy Dropout
        if 'bert' in modalities:
            self.proj_bert = nn.Sequential(
                nn.Linear(bert_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5) 
            )

        # 2. RAC Projection
        if 'rac' in modalities:
            self.proj_rac = nn.Sequential(
                nn.Linear(rac_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.ReLU()
            )

        # 3. Leaf Embedding (240 棵樹, 每棵樹對應一個 Embedding)
        if 'leaf' in modalities:
            # 這裡我們使用一個統一的 Embedding 層來處理所有樹的 index
            self.leaf_emb_layer = nn.Embedding(num_leaves+1, leaf_emb_dim)
            self.proj_leaf = nn.Sequential(
                nn.Linear(num_trees * leaf_emb_dim, mlp_hidden_dim),
                nn.LayerNorm(mlp_hidden_dim),
                nn.ReLU()
            )

        # 4. Final Classifier
        combined_dim = mlp_hidden_dim * len(modalities)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, bert_emb=None, rac_emb=None, leaf_indices=None):
        features = []
        
        # BERT 處理 + Modality Dropout
        if 'bert' in self.modalities and bert_emb is not None:
            b_feat = self.proj_bert(bert_emb)
            if self.training:
                # 隨機遮蔽 30% 的 BERT 樣本，強迫模型學習 RAC/Leaf
                mask = (torch.rand(b_feat.size(0), 1, device=b_feat.device) > 0.3).float()
                b_feat = b_feat * mask
            features.append(b_feat)

        if 'rac' in self.modalities and rac_emb is not None:
            features.append(self.proj_rac(rac_emb))

        if 'leaf' in self.modalities and leaf_indices is not None:
            # leaf_indices: [Batch, 240]
            l_emb = self.leaf_emb_layer(leaf_indices.long()) # [Batch, 240, 16]
            l_emb = l_emb.view(l_emb.size(0), -1)           # [Batch, 240*16 = 3840]
            features.append(self.proj_leaf(l_emb))

        fused_cat = torch.cat(features, dim=-1)
        return self.classifier(fused_cat)