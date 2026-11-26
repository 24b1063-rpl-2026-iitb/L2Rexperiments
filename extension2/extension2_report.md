# Extension 2: Intra-Feature Attention for Learning to Rank (Hierarchical 2D Attention)

## 1. Overview

Standard neural Learning to Rank (LTR) Transformers project raw tabular document features $x_i \in \mathbb{R}^{136}$ into a latent embedding $z_i \in \mathbb{R}^{144}$ via a single flat linear layer ($z_i = W x_i + b$). Because the second-order mixed derivatives of a linear map are identically zero, a flat projection cannot capture multiplicative feature interactions (e.g., $\text{BM25} \times \text{PageRank}$ or $\text{TitleMatch} \times \text{QueryLength}$) prior to listwise self-attention.

**Intra-Feature Attention** introduces a **Two-Level Hierarchical Attention** architecture:
1. **Level 1 (Intra-Feature Cross-Attention):** Each continuous scalar feature $x_{i,k}$ is tokenized into a $d$-dimensional vector ($e_{i,k} = x_{i,k} w_k + b_k \in \mathbb{R}^{144}$). A learnable document summary token $e_{[\text{DOC}]} \in \mathbb{R}^{1 \times 144}$ performs multi-head cross-attention ($h = 4$ heads) over all 136 feature tokens, followed by residual LayerNorm and a GELU feed-forward block ($d_{\text{ff}} = 288$).
2. **Level 2 (Inter-Document Slate Attention):** The interaction-rich document tokens $z_i$ are passed to the standard Transformer slate encoder for listwise ranking.

Using a single `[DOC]` latent query avoids quadratic $\mathcal{O}(F^2)$ feature-to-feature attention ($136 \times 136$), keeping per-document complexity linear at $\mathcal{O}(F \cdot d)$.

---

## 2. Implementation

The module is implemented as a drop-in replacement for `FCModel` (`(B, S, 136) -> (B, S, 144)`), making it modular across both standard dense ranking and sparse extensions:

* **`allrank/models/feature_attention.py`:** Implements `IntraFeatureAttention` with vectorized FT-Transformer-style continuous feature tokenization, multi-head cross-attention readout, and residual FFN.
* **`allrank/models/model.py` & `allrank/config.py`:** Integrates `IntraFeatureAttention` into `LTRModel` and exposes configuration toggles via JSON.
* **Mask-Aware Filtering Optimization:** In MSLR-WEB30K, slates are heavily zero-padded (up to $>50\%$ in training at $N=240$ and $>82\%$ in validation). `IntraFeatureAttention` flattens and filters out padded positions (`valid_indices = (~mask.view(-1)).nonzero()`), computing tokenization, cross-attention, and FFN layers strictly on valid documents before scattering results back.

Verification in `tests/test_intra_feature_attention.py` passes all 6 unit tests, including an analytical proof of non-linear multiplicative feature crossing ($|\Delta \frac{\partial z}{\partial x_0}| > 10^{-4}$ upon perturbing $x_1$), full gradient flow, and mask-aware equivalence.

---

## 3. Results (MSLR-WEB30K, 5-Fold Cross-Validation & RTX 4060 GPU)

### Ranking Quality (5-Fold CV, Slate Length $N = 240$)

| Model | Val NDCG@5 (Final) | Val NDCG@5 (Peak) | Val NDCG@10 | Val NDCG@30 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Dense (Fold 1)** | `0.5259` | `0.5280` | `0.5480` | `0.6029` |
| **Extension 2 (5-Fold Mean $\pm$ Std)** | `0.5235 ± 0.0036` | `0.5245 ± 0.0036` | `0.5430 ± 0.0025` | `0.5977 ± 0.0008` |

* Extension 2 closely matches the published Pobrotyn et al. (2020) Table 3 baseline ($0.5237 \pm 0.0033$), reaching a peak validation NDCG@5 of `0.5299` on Fold 4.
