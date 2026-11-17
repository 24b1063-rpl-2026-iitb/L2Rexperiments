import math
import torch
import torch.nn as nn
from typing import Optional


class FeatureTokenizer(nn.Module):
    """
    Continuous Feature Tokenizer (FT-Transformer style, Gorishniy et al., NeurIPS 2021).
    Maps each scalar feature x_{i,k} to a d-dimensional embedding vector:
        e_{i,k} = x_{i,k} * w_k + b_k
    where w_k, b_k in R^d are dedicated, learnable parameter vectors for feature k.
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform for linear projection weights per feature
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Uniform initialization for bias
        fan_in = self.n_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, n_features] tabular feature matrix
        :return: [N, n_features, d_model] continuous token tensor
        """
        # Vectorized continuous feature transformation:
        # x.unsqueeze(-1) has shape [N, n_features, 1]
        # self.weight has shape [n_features, d_model] (broadcasts across N)
        # self.bias has shape [n_features, d_model] (broadcasts across N)
        return x.unsqueeze(-1) * self.weight + self.bias


class LatentCrossAttention(nn.Module):
    """
    Latent Cross-Attention Readout module.
    A learnable document summary token attends across all F continuous feature tokens.
    Extracts high-order non-linear feature interactions in O(N * F * d) operations
    without the quadratic O(N * F^2 * d) overhead of full pairwise feature attention.
    """
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 288, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Learnable document summary query token
        self.doc_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.doc_token, std=0.02)

        # Multi-Head Cross-Attention: Query is [DOC], Keys & Values are feature tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Pre/Post Normalization & Residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, feature_tokens: torch.Tensor) -> torch.Tensor:
        """
        :param feature_tokens: [N, n_features, d_model]
        :return: [N, d_model] dense document representation
        """
        batch_docs = feature_tokens.size(0)
        # Expand [DOC] token across all documents in the batch
        q = self.doc_token.expand(batch_docs, -1, -1)  # [N, 1, d_model]

        # Multi-Head Cross-Attention readout
        attn_out, _ = self.cross_attn(
            query=q,
            key=feature_tokens,
            value=feature_tokens,
            need_weights=False
        )

        # Residual connection + LayerNorm
        h = self.norm1(q + self.dropout1(attn_out))

        # FFN block + Residual connection + LayerNorm
        out = self.norm2(h + self.dropout2(self.ffn(h)))

        # Squeeze query length dimension: [N, 1, d_model] -> [N, d_model]
        return out.squeeze(1)


class IntraFeatureAttention(nn.Module):
    """
    Hierarchical Level-1 Intra-Feature Attention module.
    Replaces the flat linear projection FCModel(136 -> 144) with continuous feature
    tokenization and latent cross-attention readout.

    Interface:
        Input:  [Batch, SlateLength, n_features]
        Output: [Batch, SlateLength, d_model]
    Acts as a plug-and-play drop-in replacement for FCModel.
    """
    def __init__(
        self,
        n_features: int = 136,
        d_model: int = 144,
        n_heads: int = 4,
        d_ff: int = 288,
        dropout: float = 0.1,
        chunk_size: int = 1024,
        mask_aware: bool = True
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.output_size = d_model
        self.chunk_size = chunk_size
        self.mask_aware = mask_aware

        self.tokenizer = FeatureTokenizer(n_features=n_features, d_model=d_model)
        self.readout = LatentCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: [Batch, SlateLength, n_features]
        :param mask: optional slate mask [Batch, SlateLength] where True/1 indicates padded document
        :return: [Batch, SlateLength, d_model]
        """
        b, s, f = x.shape
        x_flat = x.view(b * s, f)

        # Baseline unmasked execution (when mask is not supplied or mask_aware is False)
        if not self.mask_aware or mask is None:
            total_docs = b * s
            if total_docs <= self.chunk_size:
                tokens = self.tokenizer(x_flat)
                doc_embeddings = self.readout(tokens)
            else:
                chunks = []
                for i in range(0, total_docs, self.chunk_size):
                    chunk_x = x_flat[i : i + self.chunk_size]
                    chunk_tokens = self.tokenizer(chunk_x)
                    chunk_embed = self.readout(chunk_tokens)
                    chunks.append(chunk_embed)
                doc_embeddings = torch.cat(chunks, dim=0)
            return doc_embeddings.view(b, s, self.d_model)

        # Mask-aware filtered execution:
        # In allRank, mask == True / 1 indicates padded candidate elements.
        # Filtering out padded documents eliminates redundant tokenization, cross-attention,
        # and FFN computations (~50% in train, ~82% in val slates), cutting epoch time by >50%.
        mask_flat = mask.view(-1)
        valid_indices = (~mask_flat.bool()).nonzero(as_tuple=True)[0]

        if valid_indices.numel() == 0:
            return torch.zeros(b, s, self.d_model, device=x.device, dtype=x.dtype)

        # Process ONLY valid, non-padded documents
        x_valid = x_flat[valid_indices]
        total_valid = x_valid.size(0)

        if total_valid <= self.chunk_size:
            tokens = self.tokenizer(x_valid)
            valid_embed = self.readout(tokens)
        else:
            chunks = []
            for i in range(0, total_valid, self.chunk_size):
                chunk_x = x_valid[i : i + self.chunk_size]
                chunk_tokens = self.tokenizer(chunk_x)
                chunk_embed = self.readout(chunk_tokens)
                chunks.append(chunk_embed)
            valid_embed = torch.cat(chunks, dim=0)

        # Scatter back valid document embeddings into full [b * s, d_model] tensor
        doc_embeddings = torch.zeros(b * s, self.d_model, device=x.device, dtype=valid_embed.dtype)
        doc_embeddings[valid_indices] = valid_embed
        return doc_embeddings.view(b, s, self.d_model)
