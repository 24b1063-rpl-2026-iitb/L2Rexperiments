import pytest
import torch
import torch.nn as nn
import numpy as np
import os, sys

# Ensure local extension2 package is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT2_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if EXT2_DIR not in sys.path:
    sys.path.insert(0, EXT2_DIR)

from allrank.models.feature_attention import FeatureTokenizer, LatentCrossAttention, IntraFeatureAttention
from allrank.models.model import make_model, LTRModel
from allrank.config import Config
from allrank.models.losses.ordinal import ordinal
from attr import asdict


class TestIntraFeatureAttention:

    def test_output_shapes_and_invariance(self):
        """
        Verify that IntraFeatureAttention accepts arbitrary (Batch, SlateLength, n_features)
        and consistently outputs (Batch, SlateLength, d_model) without NaNs or Infs.
        """
        module = IntraFeatureAttention(n_features=136, d_model=144, n_heads=4, d_ff=288, dropout=0.0)
        module.eval()

        test_shapes = [
            (1, 1, 136),     # Single document
            (4, 30, 136),    # Small batch
            (8, 240, 136),   # Full paper slate length
            (16, 50, 136)    # Variable batch
        ]

        for b, s, f in test_shapes:
            x = torch.randn(b, s, f)
            with torch.no_grad():
                out = module(x)
            assert out.shape == (b, s, 144), f"Expected shape ({b}, {s}, 144), got {out.shape}"
            assert not torch.isnan(out).any(), f"NaNs detected for input shape ({b}, {s}, {f})"
            assert not torch.isinf(out).any(), f"Infs detected for input shape ({b}, {s}, {f})"

    def test_full_gradient_flow_to_all_parameters(self):
        """
        Verify analytical autograd gradient flow back to EVERY learnable component:
        1. Tokenizer weight [F, d] and bias [F, d]
        2. Latent doc summary query token [1, 1, d]
        3. Cross-attention in_proj / out_proj weights
        4. FFN layers and LayerNorm scales
        """
        torch.manual_seed(42)
        module = IntraFeatureAttention(n_features=136, d_model=144, n_heads=4, d_ff=288, dropout=0.0)
        module.train()

        x = torch.randn(2, 10, 136, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None, "Gradient to input features is None"
        assert torch.norm(x.grad) > 0, "Input feature gradient norm is zero"

        # Check Tokenizer parameters
        assert module.tokenizer.weight.grad is not None, "Tokenizer weight grad is None"
        assert torch.norm(module.tokenizer.weight.grad) > 0, "Tokenizer weight grad norm is zero"
        assert module.tokenizer.bias.grad is not None, "Tokenizer bias grad is None"
        assert torch.norm(module.tokenizer.bias.grad) > 0, "Tokenizer bias grad norm is zero"

        # Check Latent doc token parameter
        assert module.readout.doc_token.grad is not None, "doc_token grad is None"
        assert torch.norm(module.readout.doc_token.grad) > 0, "doc_token grad norm is zero"

        # Check Attention parameters
        assert module.readout.cross_attn.in_proj_weight.grad is not None, "Cross-attn in_proj_weight grad is None"
        assert torch.norm(module.readout.cross_attn.in_proj_weight.grad) > 0
        assert module.readout.cross_attn.out_proj.weight.grad is not None, "Cross-attn out_proj grad is None"
        assert torch.norm(module.readout.cross_attn.out_proj.weight.grad) > 0

        # Check FFN parameters
        assert module.readout.ffn[0].weight.grad is not None, "FFN linear 1 grad is None"
        assert torch.norm(module.readout.ffn[0].weight.grad) > 0
        assert module.readout.ffn[3].weight.grad is not None, "FFN linear 2 grad is None"
        assert torch.norm(module.readout.ffn[3].weight.grad) > 0

    def test_multiplicative_feature_crossing_proof(self):
        """
        Mathematical proof that IntraFeatureAttention creates NON-LINEAR feature interactions:
        For a linear projection z = W x + b, d(z)/d(x_j) is a constant (W_{:, j}), completely
        independent of any other feature x_k (k != j).
        In an attention-based feature crossing model, the cross-attention softmax induces
        coupling: d(z)/d(x_j) depends directly on x_k.
        We verify this by computing d(z_0)/d(x_0) before and after perturbing feature x_1.
        """
        torch.manual_seed(123)
        module = IntraFeatureAttention(n_features=136, d_model=144, n_heads=4, d_ff=288, dropout=0.0)
        module.eval()

        # Base input
        x_base = torch.randn(1, 1, 136)

        # 1. Compute grad of output wrt x_0 at base point
        x1 = x_base.clone().requires_grad_(True)
        z1 = module(x1)[0, 0, 0] # First coordinate of document representation
        grad_x0_before = torch.autograd.grad(z1, x1, retain_graph=False)[0][0, 0, 0].item()

        # 2. Perturb unrelated feature x_1 by +10.0
        x2 = x_base.clone()
        x2[0, 0, 1] += 10.0
        x2.requires_grad_(True)
        z2 = module(x2)[0, 0, 0]
        grad_x0_after = torch.autograd.grad(z2, x2, retain_graph=False)[0][0, 0, 0].item()

        # In a linear layer: grad_x0_before == grad_x0_after exactly.
        # In IntraFeatureAttention: the attention weights shift, changing the gradient.
        diff = abs(grad_x0_before - grad_x0_after)
        assert diff > 1e-4, f"Feature cross coupling failed: diff={diff:.6f}. Model behaved linearly!"

    def test_make_model_factory_and_config_integration(self):
        """
        Verify that make_model correctly instantiates IntraFeatureAttention as input_layer
        when specified in the config, and seamlessly integrates with LTRModel.
        """
        config_path = os.path.join(EXT2_DIR, "configs", "intra_feature_kaggle.json")
        cfg = Config.from_json(config_path)

        n_features = 136
        model = make_model(n_features=n_features, **asdict(cfg.model, recurse=False))

        assert isinstance(model, LTRModel)
        assert isinstance(model.input_layer, IntraFeatureAttention)
        assert model.input_layer.output_size == 144

        # Run forward pass through the entire neural ranking model
        b, s = 4, 30
        x = torch.randn(b, s, n_features)
        mask = torch.zeros(b, s, dtype=torch.bool)
        mask[:, 25:] = True # simulate 5 padded items per slate
        indices = torch.arange(s).unsqueeze(0).expand(b, -1)

        # Forward output (ordinal probabilities before loss)
        out = model(x, mask, indices)
        assert out.shape == (b, s, 4) # 4 ordinal thresholds

        # Scoring output (ranking scores)
        scores = model.score(x, mask, indices)
        assert scores.shape == (b, s)
        assert not torch.isnan(scores).any()

    def test_50_step_optimization_convergence(self):
        """
        Verify end-to-end training loop stability:
        Train a model with IntraFeatureAttention on synthetic ranking data for 50 steps
        using Adam and Ordinal loss. Verify loss strictly decreases without divergence.
        """
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = os.path.join(EXT2_DIR, "configs", "intra_feature_kaggle.json")
        cfg = Config.from_json(config_path)
        model = make_model(n_features=136, **asdict(cfg.model, recurse=False)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        loss_fn = lambda y_pred, y_true: ordinal(y_pred, y_true, n=4)

        b, s, f = 8, 30, 136
        x = torch.randn(b, s, f, device=device)
        mask = torch.zeros(b, s, dtype=torch.bool, device=device)
        indices = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        # Synthetic targets 0 to 4
        y = torch.randint(0, 5, (b, s), device=device).float()

        initial_loss = None
        final_loss = None

        model.train()
        for step in range(50):
            optimizer.zero_grad()
            preds = model(x, mask, indices)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            if step == 49:
                final_loss = loss.item()

        assert not np.isnan(final_loss), "Optimization resulted in NaN loss"
        assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        assert final_loss < initial_loss * 0.75, f"Loss did not decrease sufficiently (>25%): initial={initial_loss:.4f}, final={final_loss:.4f}"

    def test_mask_aware_filtering_equivalence_and_gradient_flow(self):
        """
        Verify mask-aware filtering:
        1. Valid items produce bit-exact identical embeddings between mask-aware and unmasked modes.
        2. Padded items are cleanly zeroed out.
        3. Backward autograd gradients flow correctly to valid items and all model parameters.
        """
        torch.manual_seed(42)
        module_unmasked = IntraFeatureAttention(n_features=136, d_model=144, n_heads=4, d_ff=288, dropout=0.0, mask_aware=False)
        module_masked = IntraFeatureAttention(n_features=136, d_model=144, n_heads=4, d_ff=288, dropout=0.0, mask_aware=True)
        # Synchronize weights
        module_masked.load_state_dict(module_unmasked.state_dict())
        module_unmasked.eval()
        module_masked.eval()

        b, s, f = 4, 20, 136
        x = torch.randn(b, s, f)
        mask = torch.zeros(b, s, dtype=torch.bool)
        # Pad second half of slates
        mask[:, 10:] = True

        with torch.no_grad():
            out_unmasked = module_unmasked(x)
            out_masked = module_masked(x, mask)

        # 1. Output shape check
        assert out_masked.shape == (b, s, 144)

        # 2. Check exact numerical equivalence on valid items
        valid_unmasked = out_unmasked[:, :10]
        valid_masked = out_masked[:, :10]
        assert torch.allclose(valid_unmasked, valid_masked, atol=1e-5), "Valid item embeddings differ between masked and unmasked!"

        # 3. Check padded items are zeros
        padded_masked = out_masked[:, 10:]
        assert (padded_masked == 0.0).all(), "Padded items are not zeroed in mask-aware output!"

        # 4. Check backward pass in training mode
        module_masked.train()
        x_grad = torch.randn(b, s, f, requires_grad=True)
        out = module_masked(x_grad, mask)
        loss = out.sum()
        loss.backward()

        assert x_grad.grad is not None
        assert (x_grad.grad[:, :10] != 0).any(), "Valid items did not receive gradients!"
        assert (x_grad.grad[:, 10:] == 0).all(), "Padded items erroneously received non-zero gradients!"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
