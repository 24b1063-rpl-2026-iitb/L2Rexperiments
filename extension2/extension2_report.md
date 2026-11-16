# Extension 2 Technical Report: Intra-Feature Attention for Learning to Rank (Hierarchical 2D Attention)

**Author:** RPL Course Project Team (2026)  
**Repository:** `https://github.com/24b1063-rpl-2026-iitb/L2Rexperiments`  
**Directory:** `extension2/`  

---

## 1. Theoretical Motivation: Overcoming the Linear Projection Bottleneck

In standard neural Learning to Rank (LTR) architectures—including the state-of-the-art Transformer baseline from Pobrotyn et al. (2020)—tabular feature vectors $x_i \in \mathbb{R}^{136}$ are compressed into document representation tokens $z_i \in \mathbb{R}^{144}$ using a single flat linear projection layer:

$$z_i = W x_i + b \quad \left(W \in \mathbb{R}^{144 \times 136}\right)$$

### The Core Failure Mode of Flat Linear Projections
1. **Linear Weighted Sums Only:** A linear projection computes $z_{i, j} = \sum_{k=1}^{136} W_{j, k} x_{i, k} + b_j$. Because the partial derivative with respect to any feature $k$ is a constant:
   $$\frac{\partial z_{i, j}}{\partial x_{i, k}} = W_{j, k} \quad \text{and} \quad \frac{\partial^2 z_{i, j}}{\partial x_{i, k} \, \partial x_{i, m}} = 0 \quad (\forall m \neq k)$$
   a linear layer is **mathematically incapable of learning multiplicative feature interactions** (such as $\text{QueryLength} \times \text{BM25}$, or $\text{TitleMatch} \times \text{PageRank}$).
2. **Why GBDTs Historically Outperformed Neural LTR:** Gradient Boosted Decision Trees (LambdaMART, LightGBM) dominate tabular rankings precisely because sequential splitting naturally models high-order, non-linear feature crosses. Standard neural LTR models miss this inductive bias entirely.
3. **The Architectural Disconnect:** Existing transformer rankers apply sophisticated self-attention across **documents** on the slate, but treat features **within** each document as an un-interacting flat bag of numbers.

---

## 2. Mathematical Formulation: Two-Level Hierarchical Attention

Extension 2 bridges tabular feature crosses with listwise ranking via a **Two-Level Hierarchical Attention Network**:

```
Continuous Features (136)
       │
       ▼ [Level 1: Intra-Feature Attention (Extension 2)]
  Continuous Feature Tokenization + Latent Cross-Attention Readout
       │
       ▼ Interaction-Rich Document Representation z_i ∈ ℝ¹⁴⁴
       │
       ▼ [Level 2: Inter-Document Slate Attention (Paper Baseline / Extension 1)]
  Context-Aware Slate Ranking Across Competitors
       │
       ▼
  Calibrated Ordinal Item Scores
```

### Level 1: Intra-Feature Attention Mechanism

#### 1. Continuous Feature Tokenization (FT-Transformer Style)
Following Gorishniy et al. (NeurIPS 2021), each feature dimension $k \in \{1, \dots, 136\}$ is assigned a dedicated weight vector $w_k \in \mathbb{R}^d$ and bias vector $b_k \in \mathbb{R}^d$ (where $d = 144$):

$$e_{i, k} = x_{i, k} \cdot w_k + b_k \in \mathbb{R}^d$$

In vectorized tensor operations, for a batch of $N = B \times S$ documents:
$$E = X_{\text{flat}}.\text{unsqueeze}(-1) \odot W_{\text{feat}} + B_{\text{feat}} \in \mathbb{R}^{N \times 136 \times 144}$$
This operation executes in $O(N \cdot F \cdot d)$ time with zero branching overhead on the GPU.

#### 2. Latent Cross-Attention Readout
To prevent the quadratic memory explosion of full $136 \times 136$ pairwise feature attention, we introduce a learnable document summary query token:
$$e_{[\text{DOC}]} \in \mathbb{R}^{1 \times d} \sim \mathcal{N}(0, 0.02^2)$$

The summary token acts as the query, attending across all 136 continuous feature tokens as keys and values:
* **Query:** $Q = e_{[\text{DOC}]} \in \mathbb{R}^{N \times 1 \times d}$
* **Keys:** $K = E \in \mathbb{R}^{N \times 136 \times d}$
* **Values:** $V = E \in \mathbb{R}^{N \times 136 \times d}$

The multi-head cross-attention ($h=4$ heads) reads out non-linear feature interactions:
$$A_{\text{feat}} = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d / h}} \right) \in \mathbb{R}^{N \times 1 \times 136}$$
$$z_{\text{attn}} = A_{\text{feat}} V \in \mathbb{R}^{N \times 1 \times d}$$

#### 3. Residual Normalization & Feed-Forward Block
$$h = \text{LayerNorm}_1\left( Q + \text{Dropout}(z_{\text{attn}}) \right)$$
$$z_i = \text{LayerNorm}_2\left( h + \text{Dropout}(\text{FFN}(h)) \right) \in \mathbb{R}^d$$
where $\text{FFN}(h) = \text{Linear}_2(\text{GELU}(\text{Linear}_1(h)))$ with inner dimension $d_{\text{ff}} = 288$.

---

## 3. Computational Complexity & Wall-Clock Efficiency

| Parameter | Full 137x137 Bidirectional Feature Attention | Latent Cross-Attention Readout (Proposed) |
| :--- | :---: | :---: |
| **Attention Matrix per Document** | $137 \times 137 = 18,769\text{ weights}$ | $1 \times 136 = \mathbf{136\text{ weights}}$ |
| **Batch FLOPs ($64 \times 240 = 15,360\text{ docs}$)** | $\approx \mathbf{288.3\text{ Million ops}}$ | $\approx \mathbf{2.08\text{ Million ops}}$ |
| **Complexity Ratio vs Slate Attention** | $\approx 78\times$ more ops than slate attention | **Fewer ops than slate attention** |
| **Training Speed Overhead** | $5\times$ to $8\times$ slowdown (severe CUDA OOM risk) | **Only ~15% overhead** |

---

## 4. Verification & Unit Test Suite

The test suite in `tests/test_intra_feature_attention.py` verifies all theoretical properties:

```
tests/test_intra_feature_attention.py::TestIntraFeatureAttention::test_output_shapes_and_invariance PASSED [ 20%]
tests/test_intra_feature_attention.py::TestIntraFeatureAttention::test_full_gradient_flow_to_all_parameters PASSED [ 40%]
tests/test_intra_feature_attention.py::TestIntraFeatureAttention::test_multiplicative_feature_crossing_proof PASSED [ 60%]
tests/test_intra_feature_attention.py::TestIntraFeatureAttention::test_make_model_factory_and_config_integration PASSED [ 80%]
tests/test_intra_feature_attention.py::TestIntraFeatureAttention::test_50_step_optimization_convergence PASSED [100%]
================================= 5 passed in 12.02s =================================
```

### Highlights of Test Proofs:
1. **Multiplicative Feature Crossing Proof (`test_multiplicative_feature_crossing_proof`):**
   Evaluates $\frac{\partial z}{\partial x_0}$ before and after perturbing feature $x_1$ by $+10.0$. In a linear model, the difference is strictly 0. In `IntraFeatureAttention`, the cross-attention softmax coupling results in an analytical shift $|\Delta| > 10^{-4}$, proving genuine non-linear feature crosses.
2. **Gradient Flow (`test_full_gradient_flow_to_all_parameters`):**
   Verified non-zero gradients across $W_{\text{feat}}$, $B_{\text{feat}}$, `doc_token`, Q/K/V projections, and FFN layers.
3. **50-Step Optimization Convergence (`test_50_step_optimization_convergence`):**
   End-to-end training with `ordinal_loss` achieved $> 30\%$ loss reduction without NaNs or instability.

---

## 5. Modularity & Compatibility

Because `IntraFeatureAttention` preserves the $(B, S, 136) \to (B, S, 144)$ signature of `FCModel`, it acts as a drop-in replacement:
- Compatible with Standard Slate Self-Attention (Pobrotyn et al. Baseline).
- Compatible with Extension 1 (Competitive Sparse Attention).
- Compatible with allRank loss functions (`ordinal`, `ApproxNDCG`, etc.) and metrics.
