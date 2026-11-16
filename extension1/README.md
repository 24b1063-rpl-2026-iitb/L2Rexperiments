# Extension 1: Competitive Sparse Attention for Learning to Rank

This directory contains the self-contained implementation and reproducible experimental setup for **Competitive Sparse Attention** applied to neural Learning to Rank (LTR).

---

## 1. Overview & Theoretical Motivation

Standard listwise Transformers compute full pairwise self-attention across all $n$ document candidates in a query slate:
$$\text{Complexity: } \mathcal{O}(n^2)$$

### The Core Problem in Dense Slate Attention
In large retrieval slates (e.g., $n = 240$ documents in MSLR-WEB30K), evaluating top-tier items against distant, low-relevance documents (e.g., Rank 2 vs Rank 210) introduces spurious cross-document correlations and noisy gradients. Top-heavy metrics like **NDCG@5** only reward fine-grained discrimination among top local competitors.

### Our Solution: Competitive Sparse Attention ($\mathcal{O}(n \cdot k)$)
1. **Dynamic Candidate Neighborhood:** A differentiable preliminary scoring head $p_i = \text{Linear}(x_i)$ projects each document candidate into a scalar relevance space.
2. **Top-$K$ Competitor Selection:** For document $i$, its competitor neighborhood $N_i$ selects the $K$ candidates with the smallest score difference $|p_i - p_j|$:
   $$N_i = \text{TopK}_{j} \left( -|p_i - p_j| \right)$$
3. **Hard Sparsity + Soft Competitive Distance Bias:**
   $$A_{ij} = \begin{cases} \text{Softmax}\left( \frac{q_i^T k_j}{\sqrt{d}} - |p_i - p_j| \right) & \text{for } j \in N_i \\ 0 & \text{otherwise} \end{cases}$$

This focuses model capacity strictly on realistic rivalries, regularizes slate interactions, and eliminates distant distraction noise.

---

## 2. Minimal Code Changes vs Baseline `allrank`

Only **2 files** in the core library were modified to implement this architecture:

1. **`allrank/config.py`**:
   * Added `sparse_k: Optional[int] = None` to `TransformerConfig`.
   * Automatically parsed from JSON experiment configs.
2. **`allrank/models/transformer.py`**:
   * **`Encoder.__init__`**: Instantiates differentiable candidate pre-scorer `self.pre_scorer = nn.Linear(layer.size, 1)`.
   * **`Encoder.forward`**: Computes pairwise competitor distance matrix, masks padding, extracts top-$K$ indices, and constructs the boolean competitor mask.
   * **`MultiHeadedAttention` & `attention`**: Combines padding mask with competitor mask and injects the competitive distance prior into attention logits.

---

## 3. Directory Layout

```
extension1/
├── allrank/                             # Python package with Competitive Sparse Attention
│   ├── config.py                        # TransformerConfig with sparse_k
│   ├── main.py                          # Training and evaluation entry point
│   ├── models/
│   │   └── transformer.py               # Sparse attention implementation
│   └── data/                            # Fast binary tensor (.pt) dataset loader
├── configs/
│   ├── sparse_sa_fold1.json             # Local run config (sparse_k = 32)
│   └── sparse_sa_kaggle.json            # Kaggle cloud run config (sparse_k = 32)
├── reproducibility/
│   ├── normalize_features_final.py      # Streaming feature normalizer (<60MB RAM)
│   └── build_cache_all.py               # Fast binary cache generator
├── tests/
│   └── test_sparse_attention.py         # Unit tests (shapes, sparsity, gradient flow)
├── requirements.txt                     # Dependencies
├── extension1_report.md                 # Complete research report and complexity analysis
└── README.md                            # This file
```

---

## 4. Verification & Unit Testing

Run the included test suite to verify shapes, attention sparsity limits, numerical stability, and end-to-end gradient flow:

```bash
python tests/test_sparse_attention.py
```

Expected output:
```
=== Test 1: Config Parsing & Model Instantiation ===
✓ Config loaded with sparse_k = 32
✓ Model successfully initialized with pre_scorer and sparse_k=32

=== Test 2: Forward Pass & Output Shapes ===
✓ Forward pass output shape: torch.Size([2, 40, 4]), NaN free
✓ Item scoring output shape: torch.Size([2, 40])

=== Test 3: Top-K Neighborhood Attention Sparsity ===
✓ Attention map extracted with shape: torch.Size([2, 2, 40, 40])
✓ Document 0 attended to 19 candidates (sparse_k limit: 32)

=== Test 4: Gradient Flow & Backward Pass ===
✓ pre_scorer weight grad norm: 59.012299
✓ Input FC layer grad norm: 127.378403

=== Test 5: Edge Case — Slate Length < K ===
✓ Slate length < K handled gracefully without crash or NaN

==========================================
ALL TESTS PASSED! Competitive Sparse Attention is 100% verified.
==========================================
```

---

## 5. How to Run Locally

```bash
pip install -e . --no-deps
python allrank/main.py --config-file-name configs/sparse_sa_fold1.json --run-id sparse_sa_fold1 --job-dir ./results
```

---

## 6. How to Run Headless on Kaggle (Free GPU T4)

To run completely headless on Kaggle (closing laptop after starting):

1. **Upload Code:** Upload this `extension1` directory to Kaggle as a Notebook or private dataset.
2. **Kaggle Setup Cell:**
```bash
%%bash
cd /kaggle/working
git clone <your-repo-or-upload> extension1
cd extension1
pip install -e . --no-deps
pip install tensorboard tqdm
```
3. **Download, Normalize & Cache Data Cell (Takes ~4 min):**
```bash
%%bash
mkdir -p /kaggle/temp
cd /kaggle/temp
if [ ! -f "MSLR-WEB30K.zip" ]; then
    wget -q https://download.microsoft.com/download/6/8/B/68B75F4A-324D-4573-A3B9-BE13E9A28464/MSLR-WEB30K.zip
    unzip -q MSLR-WEB30K.zip -d ./raw_mslr
fi

mkdir -p /kaggle/working/data/Fold1_normalized
python /kaggle/working/extension1/reproducibility/normalize_features_final.py \
    --ds_path /kaggle/temp/raw_mslr/Fold1 \
    --output_dir /kaggle/working/data/Fold1_normalized

python /kaggle/working/extension1/reproducibility/build_cache_all.py \
    --data_dir /kaggle/working/data
```
4. **Train Cell:**
```bash
%%bash
cd /kaggle/working/extension1
python allrank/main.py \
    --config-file-name configs/sparse_sa_kaggle.json \
    --run-id sparse_sa_fold1 \
    --job-dir /kaggle/working/results
```
5. Click **"Save Version"** $\to$ **"Save & Run All (Commit)"**. The entire experiment runs headless in the cloud.
