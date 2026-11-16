# Learning to Rank (L2R) Research Extensions

This repository contains experimental extensions to neural Learning to Rank architectures based on Transformer self-attention.

## Repository Structure

* **`extension1/`**: **Competitive Sparse Attention ($O(n \cdot k)$)**
  * Restricts self-attention to dynamically selected local competitor neighborhoods.
  * Solves slate-level quadratic scaling and eliminates noise from distant candidate pairs.
  * Includes local configs, Kaggle configs, test suites, and reproducibility scripts.

* **`extension2/`**: *(Coming Soon)* **Intra-Feature Attention (Hierarchical 2D Attention)**
  * Continuous feature tokenization and latent cross-attention readout over tabular features.
  * Models high-order multiplicative feature crosses before listwise slate ranking.
