# Extension 1: Competitive Sparse Attention for Learning to Rank

## 1. Overview & Theoretical Formulation

In Learning to Rank (LTR) architectures using Transformer encoders (such as *allRank*), self-attention models cross-document dependencies across the entire query slate.

### Complexity Comparison
* **Standard Attention Complexity:** $\mathcal{O}(n^2)$ across all document pairs in a query slate of length $n$.
* **Competitive Sparse Attention Complexity:** $\mathcal{O}(n \cdot k)$ or $\mathcal{O}(n \cdot (k + \log n))$ where $k \ll n$ by restricting attention to a document's local competitor neighborhood.

### Algorithmic Formulation
1. **Preliminary Relevance Score:**  
   Each document candidate $i$ is assigned an initial scalar score $p_i$ (e.g., via a 1D linear projection head from initial document embeddings or a strong pointwise feature).
2. **Local Neighborhood Definition:**  
   For document $i$, its local competitor neighborhood $N_i$ is dynamically defined as:
   $$N_i = \text{TopK}_{j} \left( -|p_i - p_j| \right)$$
3. **Sparse Attention Weighting:**  
   The attention weights $A_{ij}$ are computed exclusively over the local neighborhood:
   $$A_{ij} = \begin{cases} \text{Softmax}\left(\frac{q_i^T k_j}{\sqrt{d}}\right) & \text{for } j \in N_i \\ 0 & \text{otherwise} \end{cases}$$

---

## 2. Implementation Difficulty

**Difficulty Level:** Moderate (approx. **5/10 to 6/10**).

The logic integrates cleanly into PyTorch without requiring low-level CUDA kernel modifications:
* **Pointwise Pre-scoring:** A lightweight linear layer projects document representations $x \in \mathbb{R}^{B \times n \times d}$ to $p \in \mathbb{R}^{B \times n \times 1}$.
* **Pairwise Distance Computation:** Pairwise distance $|p_i - p_j|$ is calculated via broadcasting in shape $(B, n, n)$.
* **Top-K Masking:** `torch.topk` extracts the $k$ nearest candidate indices, creating a boolean neighborhood mask that is combined with the standard padding mask.

---

## 3. Files Requiring Modification

Only **3 files** in the `allRank` codebase need modification:

1. **`allrank/models/transformer.py`:**
   * Extend the `attention()` function to accept a competitor mask (setting attention scores to $-\infty$ outside $N_i$).
   * Update `MultiHeadedAttention` and `EncoderLayer` to pass this neighborhood constraint down to each head.
2. **`allrank/models/model.py`:**
   * In `LTRModel`, introduce an initial pointwise scoring head (`p = self.pre_scorer(x)`) before passing embeddings to the encoder.
3. **`allrank/config.py`:**
   * Add a `sparse_k: Optional[int] = None` field to `TransformerConfig` so sparse attention can be enabled, disabled, and tuned directly from JSON experiment configs without affecting standard runs.

---

## 4. Performance & Speed Analysis

### Theoretical vs. Practical Speed on RTX 4060 GPU

* **Theoretical Savings:**  
  For a typical slate length of $n = 240$ with $k = 32$, theoretical attention comparisons are reduced by **$\approx 85\%$** ($\frac{32}{240} \approx 0.13$).
* **Wall-Clock Speed Reality on GPU:**  
  For $n = 240$, the dense matrix multiplication is already extremely small for modern Tensor Cores (taking $< 0.1\text{ ms}$).
  * **Dense Masking (`masked_fill`):** Wall-clock training time per epoch will remain roughly identical because the full $240 \times 240$ matrix multiplication is still executed before masking.
  * **Gathered Sparse Attention (`torch.gather`):** Index lookup and non-contiguous memory gather overhead on small matrices often offset the theoretical compute reduction in pure PyTorch unless specialized block-sparse kernels (e.g., FlashAttention / Triton) are used.
  * **Large Slate Benefit:** If slate sizes expand to $n = 1,000$–$5,000$ documents (e.g., large-scale candidate re-ranking), computational savings become significant.

---

## 5. Primary Value: Ranking Accuracy & Noise Reduction

The primary motivation and expected benefit of this extension is **ranking quality (NDCG@5)** rather than raw wall-clock acceleration:

1. **Spurious Correlation & Noise Elimination:**  
   In MSLR-WEB30K, each query slate contains up to 240 documents, but NDCG@5 evaluates only the top 5 positions. When top-tier documents attend to distant, irrelevant documents (e.g. rank 2 vs rank 210), spurious cross-document correlations inject noise into gradients.
2. **Focused Representation Capacity:**  
   Restricting context to $N_i$ forces the Transformer's self-attention mechanism to concentrate its expressive power exclusively on fine-grained margins between close competitors (e.g., distinguishing rank 2 from rank 3).
3. **Structural Regularization:**  
   Localizing the attention field acts as an inductive bias and regularizer, preventing the ranker from overfitting to arbitrary global combinations of documents within training slates.
