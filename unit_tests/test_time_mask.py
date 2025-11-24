import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from neural_decoder.neural_decoder_trainer import apply_time_mask_batch
import torch

def test_time_mask_disabled_no_change():
    """If n_masks=0 or max_mask_frac=0, X should be returned unchanged."""
    X = torch.ones(2, 10, 3)  # (B=2, T=10, C=3)
    X_len = torch.tensor([7, 5]) 

    X_orig = X.clone()

    # Case 1: n_masks = 0
    X_masked = apply_time_mask_batch(X.clone(), X_len, n_masks=0, max_mask_frac=0.5)
    assert torch.allclose(X_masked, X_orig)

    # Case 2: max_mask_frac = 0
    X_masked = apply_time_mask_batch(X.clone(), X_len, n_masks=5, max_mask_frac=0.0)
    assert torch.allclose(X_masked, X_orig)


def test_time_mask_respects_sequence_lengths():
    """
    Time masking should only modify entries in [0:X_len[b]) for each sequence.
    Padded region (X_len[b]:) must remain unchanged.
    """
    torch.manual_seed(0)  # make randomness repeatable(ish) for this test

    B, T, C = 2, 10, 3
    X = torch.ones(B, T, C)  # start with all ones
    X_len = torch.tensor([6, 8])  # sequence 0: real length 6, seq 1: length 8

    X_masked = apply_time_mask_batch(X.clone(), X_len, n_masks=5, max_mask_frac=0.5)

    # 1) padded region must stay as ones
    # seq 0: positions 6..9 are padding
    assert torch.all(X_masked[0, 6:, :] == 1.0)
    # seq 1: positions 8..9 are padding
    assert torch.all(X_masked[1, 8:, :] == 1.0)

    # 2) some zeros must appear in the real region
    # because of randomness, we cannot assert how many, but at least one zero exists in the real part for each seq
    assert torch.any(X_masked[0, :6, :] == 0.0)
    assert torch.any(X_masked[1, :8, :] == 0.0)


def test_time_mask_does_not_change_shape_or_device():
    """Masking should keep the same shape and stay on the same device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.ones(3, 12, 4, device=device)
    X_len = torch.tensor([12, 10, 7], device=device)

    X_masked = apply_time_mask_batch(X, X_len, n_masks=3, max_mask_frac=0.25)

    assert X_masked.shape == X.shape
    assert X_masked.device == X.device
