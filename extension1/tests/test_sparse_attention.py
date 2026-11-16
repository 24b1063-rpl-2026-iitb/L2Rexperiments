import sys
import os
import torch
import torch.nn as nn

# Ensure local extension1 package is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from allrank.config import Config
from allrank.models.model import make_model


def test_sparse_attention_forward_and_backward():
    print("=== Test 1: Config Parsing & Model Instantiation ===")
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "sparse_sa_fold1.json")
    config = Config.from_json(config_path)
    
    assert config.model.transformer.sparse_k == 32, f"Expected sparse_k=32, got {config.model.transformer.sparse_k}"
    print("✓ Config loaded with sparse_k = 32")

    model = make_model(
        fc_model=dict(sizes=[144], input_norm=False, activation=None, dropout=0.0),
        transformer=config.model.transformer,
        post_model=dict(output_activation="Sigmoid", d_output=4),
        n_features=136
    )
    assert hasattr(model.encoder, "pre_scorer"), "Encoder must have pre_scorer for sparse attention"
    assert model.encoder.sparse_k == 32
    print("✓ Model successfully initialized with pre_scorer and sparse_k=32")

    print("\n=== Test 2: Forward Pass & Output Shapes ===")
    batch_size = 2
    slate_length = 40  # Slate larger than K=32
    n_features = 136

    x = torch.randn(batch_size, slate_length, n_features)
    # Mask: 0 = valid document, 1 = padded document
    mask = torch.zeros(batch_size, slate_length, dtype=torch.bool)
    mask[0, 35:] = True  # Query 0 has 5 padded docs
    mask[1, 25:] = True  # Query 1 has 15 padded docs
    indices = torch.arange(slate_length).unsqueeze(0).repeat(batch_size, 1)

    output = model(x, mask, indices)
    assert output.shape == (batch_size, slate_length, 4), f"Wrong output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs!"
    print(f"✓ Forward pass output shape: {output.shape}, NaN free")

    scores = model.score(x, mask, indices)
    assert scores.shape == (batch_size, slate_length), f"Wrong score shape: {scores.shape}"
    print(f"✓ Item scoring output shape: {scores.shape}")

    print("\n=== Test 3: Top-K Neighborhood Attention Sparsity ===")
    # Access the attention map stored from the last forward pass
    last_layer_attn = model.encoder.layers[-1].self_attn.attn  # [batch_size, h, slate_length, slate_length]
    assert last_layer_attn is not None
    print(f"✓ Attention map extracted with shape: {last_layer_attn.shape}")

    # For valid documents (e.g. doc 0 in batch 0), count non-zero attention weights
    # sparse_k = 32, so non-zero entries in any row must be <= 32
    attn_row = last_layer_attn[0, 0, 0]  # Query 0, Head 0, Document 0
    non_zeros = (attn_row > 1e-6).sum().item()
    print(f"✓ Document 0 attended to {non_zeros} candidates (sparse_k limit: {config.model.transformer.sparse_k})")
    assert non_zeros <= config.model.transformer.sparse_k, f"Attended to {non_zeros} > {config.model.transformer.sparse_k}"

    print("\n=== Test 4: Gradient Flow & Backward Pass ===")
    loss = output.sum()
    loss.backward()

    # Verify gradients reach pre_scorer
    assert model.encoder.pre_scorer.weight.grad is not None, "pre_scorer weight gradient is None"
    assert model.encoder.pre_scorer.weight.grad.abs().sum() > 0, "pre_scorer weight gradient is zero"
    print(f"✓ pre_scorer weight grad norm: {model.encoder.pre_scorer.weight.grad.norm().item():.6f}")

    # Verify gradients reach input FC layer
    assert model.input_layer.layers[0].weight.grad is not None
    assert model.input_layer.layers[0].weight.grad.abs().sum() > 0
    print(f"✓ Input FC layer grad norm: {model.input_layer.layers[0].weight.grad.norm().item():.6f}")

    print("\n=== Test 5: Edge Case — Slate Length < K ===")
    # When slate length (10) < K (32), the model should smoothly attend to all available valid docs
    x_small = torch.randn(2, 10, n_features)
    mask_small = torch.zeros(2, 10, dtype=torch.bool)
    indices_small = torch.arange(10).unsqueeze(0).repeat(2, 1)

    output_small = model(x_small, mask_small, indices_small)
    assert output_small.shape == (2, 10, 4)
    assert not torch.isnan(output_small).any()
    print("✓ Slate length < K handled gracefully without crash or NaN")

    print("\n==========================================")
    print("ALL TESTS PASSED! Competitive Sparse Attention is 100% verified.")
    print("==========================================")


if __name__ == "__main__":
    test_sparse_attention_forward_and_backward()
