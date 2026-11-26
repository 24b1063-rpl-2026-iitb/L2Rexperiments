# Extension 1: Competitive Sparse Attention for Learning to Rank

## 1. Overview

In Transformer-based Learning to Rank (LTR) models, standard self-attention computes pairwise interactions across all $\mathcal{O}(N^2)$ document pairs in a query slate of length $N$. However, top-position ranking metrics (such as NDCG@5) depend primarily on fine-grained distinctions among closely competing documents rather than distant, irrelevant items.

**Competitive Sparse Attention** restricts each document's receptive field to its $K$ nearest competitors ($K \ll N$) in score space, reducing attention complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \cdot K)$:

1. **Preliminary Relevance Prior:** A linear `pre_scorer` projects each document embedding $x_i \in \mathbb{R}^d$ to a 1D scalar score $p_i \in \mathbb{R}$.
2. **1D Competitor Neighborhood:** Documents are sorted by $p_i$ in $\mathcal{O}(N \log N)$ time, turning nearest-competitor selection into a contiguous local sliding window of width $K$.
3. **Soft Distance Bias & Sparse Attention:** Each document attends strictly to its $K$ window neighbors with an added 1D distance bias $-|p_i - p_j|$:
   $$\text{Score}_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} - |p_i - p_j|, \quad j \in \text{Window}_K(i)$$

---

## 2. Implementation

Only **`allrank/models/transformer.py`** (along with `sparse_k` configuration in `allrank/config.py`) is modified, keeping all model parameter names compatible with standard checkpoints:

* **`Encoder.forward`:** Computes preliminary scalar priors $p = \text{pre\_scorer}(x)$ once per slate and passes $p$ and `sparse_k` to each encoder block.
* **`fast_sliding_window_attention`:**
  1. Sorts $Q, K, V$, and $p$ in 1D along the slate dimension (`torch.argsort` + `torch.gather`).
  2. Extracts contiguous $K$-sized neighbor windows via `F.pad` and `.unfold()` without materializing any $N \times N$ tensor.
  3. Adds the 1D soft distance bias $-|p_i - p_j|$ and masks out-of-bounds boundary pads and padded documents (`-inf`).
  4. Computes softmax attention over $N \times K$ pairs strictly and restores the original document order via inverse permutation.

Verification in `tests/test_sparse_attention.py` passes all 5 unit tests (output shapes, sparsity bounds $\le K$, gradient flow to `pre_scorer`, and $N < K$ edge cases).

---

## 3. Results (MSLR-WEB30K, 5-Fold Cross-Validation & RTX 4060 GPU)

### Ranking Quality (5-Fold CV, Slate Length $N = 240$)

| Model | Val NDCG@5 (Final) | Val NDCG@5 (Peak) | Val NDCG@10 | Val NDCG@30 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Dense (Fold 1)** | `0.5259` | `0.5280` | `0.5480` | `0.6029` |
| **Extension 1 ($K = 16$, 5-Fold Mean $\pm$ Std)** | `0.5158 ± 0.0036` | `0.5173 ± 0.0032` | `0.5356 ± 0.0026` | `0.5890 ± 0.0015` |
| **Extension 1 ($K = 32$, 5-Fold Mean $\pm$ Std)** | `0.5206 ± 0.0025` | `0.5223 ± 0.0020` | `0.5405 ± 0.0018` | `0.5941 ± 0.0007` |

* Increasing $K$ from 16 to 32 recovers ranking performance to within $\approx 0.005$ of the dense baseline while attending to only $\sim 13\%$ of the slate.

### Inference Latency Scaling (NVIDIA RTX 4060 GPU, $K = 32$)

| Candidate Slate Length ($N$) | Dense Baseline | Naive Masked-Fill | 1D Sliding Window (Ours) | Speedup vs. Naive | Speedup vs. Dense |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **240** | `3.27 ms` | `3.78 ms` | `3.54 ms` | $1.07\times$ | $0.92\times$ |
| **1,440** *(Crossover)* | `4.33 ms` | `5.62 ms` | **`3.51 ms`** | $1.60\times$ | **$1.23\times$** |
| **2,880** | `14.98 ms` | `27.81 ms` | **`6.57 ms`** | $4.24\times$ | **$2.28\times$** |
| **4,320** | `31.42 ms` | `61.50 ms` | **`9.45 ms`** | **$6.51\times$** | **$3.32\times$** |

* Replacing eager $N \times N$ masking with the 1D-sorted sliding window eliminates the $18.66\text{M}$-element sorting bottleneck at $N = 4,320$, yielding a **$3.32\times$ speedup over dense attention** and a **$6.51\times$ speedup over naive sparsity**.
