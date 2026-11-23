"""
Code in this file was adapted from "The Annotated Transformer" by Harvard NLP.
http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from allrank.config import PositionalEncoding
from allrank.models.positional import _make_positional_encoding


def clones(module, N):
    """
    Creation of N identical layers.
    :param module: module to clone
    :param N: number of copies
    :return: nn.ModuleList of module copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Stack of Transformer encoder blocks with positional encoding and competitive sparse attention.
    """
    def __init__(self, layer, N, position, sparse_k: Optional[int] = None):
        """
        :param layer: single building block to clone
        :param N: number of copies
        :param position: positional encoding module
        :param sparse_k: top-k competitor neighborhood size (None for standard dense attention)
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.position = position
        self.sparse_k = sparse_k
        if self.sparse_k is not None:
            self.pre_scorer = nn.Linear(layer.size, 1)
        else:
            self.pre_scorer = None

    def forward(self, x, mask, indices):
        """
        Forward pass through each block of the Transformer.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        if self.position:
            x = self.position(x, mask, indices)

        p = None
        if self.sparse_k is not None:
            # 1) Compute preliminary scalar relevance scores
            p = self.pre_scorer(x).squeeze(-1)  # [batch_size, slate_length]

        mask = mask.unsqueeze(-2)
        for layer in self.layers:
            x = layer(x, mask, p=p, sparse_k=self.sparse_k)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    """
    def __init__(self, features, eps=1e-6):
        """
        :param features: shape of normalized features
        :param eps: epsilon used for standard deviation
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # type: ignore
        self.b_2 = nn.Parameter(torch.zeros(features))  # type: ignore
        self.eps = eps

    def forward(self, x):
        """
        Forward pass through the layer normalization.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :return: normalized input of shape [batch_size, slate_length, output_dim]
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    Please not that for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        """
        :param size: number of input/output features
        :param dropout: dropout probability
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Forward pass through the sublayer connection module, applying the residual connection to any sublayer with the same size.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param sublayer: layer through which to pass the input prior to applying the sum
        :return: output of shape [batch_size, slate_length, output_dim]
        """
        return x + self.dropout(
            sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder block made of self-attention and feed-forward layers with residual connections.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: input/output size of the encoder block
        :param self_attn: self-attention layer
        :param feed_forward: feed-forward layer
        :param dropout: dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, p=None, sparse_k=None):
        """
        Forward pass through the encoder block.
        :param x: input of shape [batch_size, slate_length, self.size]
        :param mask: padding mask of shape [batch_size, 1, slate_length]
        :param p: preliminary relevance prior scores [batch_size, slate_length]
        :param sparse_k: competitor window size
        :return: output of shape [batch_size, slate_length, self.size]
        """
        x = self.sublayer[0](x, lambda inp: self.self_attn(inp, inp, inp, mask=mask, p=p, sparse_k=sparse_k))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    Standard Scaled Dot Product Attention computation.
    :param query: query set of shape [batch_size, n_attention_heads, slate_size, attention_dim]
    :param key: key set of shape [batch_size, n_attention_heads, slate_size, attention_dim]
    :param value: value set of shape [batch_size, n_attention_heads, slate_size, attention_dim]
    :param mask: padding mask of shape [batch_size, 1, 1, slate_length]
    :param dropout: dropout probability
    :return: attention scores of shape [batch_size, n_attention_heads, slate_size, attention_dim]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 1, float("-inf"))

    p_attn = F.softmax(scores, dim=-1)
    del scores
    p_attn = torch.nan_to_num(p_attn, nan=0.0)
    attn_weights = p_attn.detach()
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), attn_weights


def fast_sliding_window_attention(query, key, value, p, mask=None, K=32, dropout=None):
    """
    1D-Sorted Sliding Window Sparse Attention computation with O(N * K) complexity.
    :param query: query set of shape [batch_size, n_attention_heads, slate_size, attention_dim]
    :param key: key set of shape [batch_size, n_attention_heads, slate_size, attention_dim]
    :param value: value set of shape [batch_size, n_attention_heads, slate_size, attention_dim]
    :param p: 1D relevance prior of shape [batch_size, slate_size]
    :param mask: padding mask of shape [batch_size, 1, slate_size] or [batch_size, slate_size]
    :param K: competitor window size
    :param dropout: dropout probability
    :return: output tensor [batch_size, n_attention_heads, slate_size, attention_dim], attention weights
    """
    B, H, N, D_k = query.shape
    K = min(K, N)
    pad_left = K // 2
    pad_right = K - 1 - pad_left

    # 1. 1D sort of prior p (O(N log N) < 0.02 ms)
    sort_indices = torch.argsort(p, dim=-1)  # [B, N]
    sort_idx_exp = sort_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, N, D_k)
    inv_indices = torch.argsort(sort_indices, dim=-1).unsqueeze(1).unsqueeze(-1).expand(B, H, N, D_k)

    q_s = torch.gather(query, dim=2, index=sort_idx_exp)
    k_s = torch.gather(key, dim=2, index=sort_idx_exp)
    v_s = torch.gather(value, dim=2, index=sort_idx_exp)
    p_s = torch.gather(p, dim=1, index=sort_indices)

    # 2. Window extraction via unfold
    pad_k = (0, 0, pad_left, pad_right)
    k_padded = F.pad(k_s, pad_k, mode='replicate')
    v_padded = F.pad(v_s, pad_k, mode='replicate')

    k_neighbors = k_padded.unfold(dimension=2, size=K, step=1)  # [B, H, N, D_k, K]
    v_neighbors = v_padded.unfold(dimension=2, size=K, step=1).transpose(-1, -2)  # [B, H, N, K, D_k]

    # 3. Attention scores: [B, H, N, K]
    scores = torch.matmul(q_s.unsqueeze(-2), k_neighbors).squeeze(-2) / math.sqrt(D_k)

    # 4. Extract and add 1D Soft Distance Bias (-|p_i - p_j|)
    p_padded = F.pad(p_s, (pad_left, pad_right), mode='replicate')
    p_neighbors = p_padded.unfold(dimension=1, size=K, step=1)  # [B, N, K]
    dist_bias = -torch.abs(p_s.unsqueeze(-1) - p_neighbors).unsqueeze(1)  # [B, 1, N, K]
    scores = scores + dist_bias

    # 5. Boundary Masking (-1e9 over padded out-of-bounds positions)
    pad_bool = torch.zeros(B, N, device=query.device, dtype=torch.bool)
    pad_mask = F.pad(pad_bool, (pad_left, pad_right), value=True).unfold(1, K, 1).unsqueeze(1)
    scores = scores.masked_fill(pad_mask, float("-inf"))

    # 6. Padding candidate masking (if query has padded docs)
    if mask is not None:
        mask_2d = (mask == 1).squeeze(1) if mask.dim() == 3 else (mask == 1)
        mask_s = torch.gather(mask_2d, dim=1, index=sort_indices)
        mask_neighbors = F.pad(mask_s, (pad_left, pad_right), value=True).unfold(1, K, 1).unsqueeze(1)
        scores = scores.masked_fill(mask_neighbors, float("-inf"))

    # 7. Softmax and restore original document ordering
    p_attn = F.softmax(scores, dim=-1)
    p_attn = torch.nan_to_num(p_attn, nan=0.0)
    attn_weights = p_attn.detach()
    if dropout is not None:
        p_attn = dropout(p_attn)

    out_s = torch.matmul(p_attn.unsqueeze(-2), v_neighbors).squeeze(-2)  # [B, H, N, D_k]
    out = torch.gather(out_s, dim=2, index=inv_indices)  # [B, H, N, D_k]
    return out, attn_weights


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention block.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: number of attention heads
        :param d_model: input/output dimensionality
        :param dropout: dropout probability
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, p=None, sparse_k=None):
        """
        Forward pass through the multi-head attention block.
        :param query: query set of shape [batch_size, slate_size, self.d_model]
        :param key: key set of shape [batch_size, slate_size, self.d_model]
        :param value: value set of shape [batch_size, slate_size, self.d_model]
        :param mask: padding mask of shape [batch_size, 1, slate_length]
        :param p: preliminary relevance prior scores [batch_size, slate_length]
        :param sparse_k: competitor window size
        :return: output of shape [batch_size, slate_size, self.d_model]
        """
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        if sparse_k is not None and p is not None:
            x, self.attn = fast_sliding_window_attention(
                query, key, value, p=p, mask=mask, K=sparse_k, dropout=self.dropout
            )
        else:
            if mask is not None:
                mask = mask.unsqueeze(1)
            x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Feed-forward block.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: input/output dimensionality
        :param d_ff: hidden dimensionality
        :param dropout: dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the feed-forward block.
        :param x: input of shape [batch_size, slate_size, self.d_model]
        :param炭: output of shape [batch_size, slate_size, self.d_model]
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def make_transformer(N=6, d_ff=2048, h=8, dropout=0.1, n_features=136,
                     positional_encoding: Optional[PositionalEncoding] = None,
                     sparse_k: Optional[int] = None):
    """
    Helper function for instantiating Transformer-based Encoder with optional competitive sparse attention.
    :param N: number of Transformer blocks
    :param d_ff: hidden dimensionality of the feed-forward layer in the Transformer block
    :param h: number of attention heads
    :param dropout: dropout probability
    :param n_features: number of input/output features of the feed-forward layer
    :param positional_encoding: config.PositionalEncoding object containing PE config
    :param sparse_k: top-k competitor neighborhood size
    :return: Transformer-based Encoder with given hyperparameters
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, n_features, dropout)

    ff = PositionwiseFeedForward(n_features, d_ff, dropout)
    position = _make_positional_encoding(n_features, positional_encoding)
    return Encoder(EncoderLayer(n_features, c(attn), c(ff), dropout), N, position, sparse_k=sparse_k)
