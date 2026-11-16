# Extension 2: Intra-Feature Attention (Hierarchical 2D Attention for LTR)

This repository contains the standalone, fully executable implementation of **Extension 2: Intra-Feature Attention for Learning to Rank**.

---

## Overview

In standard Learning to Rank models (including Pobrotyn et al., 2020), document features $x_i \in \mathbb{R}^{136}$ are compressed into document embeddings $z_i \in \mathbb{R}^{144}$ using a single flat linear projection ($z_i = W x_i + b$). This imposes a strict linear bottleneck: the model cannot compute multiplicative feature interactions.

**Extension 2** replaces the linear layer with a **Two-Level Hierarchical Attention Network**:
- **Level 1 (Intra-Feature Attention):** Uses continuous feature tokenization ($e_{ik} = x_{ik} \cdot w_k + b_k$) and a **Latent Cross-Attention Readout** where a learnable `[DOC]` summary token attends across all 136 continuous feature tokens.
- **Level 2 (Inter-Document Attention):** Feeds interaction-rich document embeddings into the 4-layer slate Transformer for listwise context-aware ranking.

---

## Directory Structure

```
extension2/
├── allrank/
│   ├── models/
│   │   ├── feature_attention.py  # FeatureTokenizer, LatentCrossAttention, IntraFeatureAttention
│   │   ├── model.py             # make_model factory supporting intra_feature
│   │   ├── transformer.py       # Slate Transformer
│   │   └── metrics.py           # Ranking metrics (NDCG)
│   ├── config.py                # Config dataclasses with intra_feature support
│   └── training/train_utils.py  # Headless clean training loop
├── configs/
│   ├── intra_feature_kaggle.json  # 5-fold training config for Kaggle
│   └── intra_feature_sa.json      # Local training config
├── tests/
│   └── test_intra_feature_attention.py  # 5 comprehensive unit tests
├── reproducibility/
│   ├── normalize_features_final.py
│   └── build_cache_all.py
├── extension2_report.md          # Full theoretical and architectural report
├── README.md
└── setup.py
```

---

## Running the Unit Tests

```bash
python -m pytest tests/test_intra_feature_attention.py -v
```

All 5 tests verify:
1. Output shape invariance across batch and slate dimensions.
2. Gradient flow back to all tokenizer and attention weights.
3. Mathematical proof of non-linear multiplicative feature crossing ($\Delta \nabla \neq 0$).
4. Seamless integration with `make_model` and `LTRModel`.
5. 50-step end-to-end optimization convergence with ordinal loss.

---

## Configuration

To activate Intra-Feature Attention, add the `intra_feature` dictionary to `fc_model` in the experiment JSON:

```json
{
  "model": {
    "fc_model": {
      "sizes": [144],
      "input_norm": false,
      "activation": null,
      "dropout": 0.0,
      "intra_feature": {
        "n_heads": 4,
        "d_ff": 288,
        "dropout": 0.1
      }
    },
    "transformer": {
      "N": 4,
      "d_ff": 512,
      "h": 2,
      "dropout": 0.4
    },
    "post_model": {
      "output_activation": "Sigmoid",
      "d_output": 4
    }
  }
}
```
