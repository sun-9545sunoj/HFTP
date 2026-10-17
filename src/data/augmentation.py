"""
LOB-Specific Data Augmentation for Limit Order Book Sequence Models.

Implements novel augmentation strategies designed specifically for normalized
LOB feature matrices to reduce overfitting and improve out-of-sample generalization.

Augmentation Techniques:
    1. PriceNoise:      Gaussian noise on price features (bid/ask levels)
    2. VolumeScaling:   Random multiplicative scaling on volume features
    3. LevelDropout:    Randomly zero out entire LOB depth levels (simulate missing quotes)
    4. FeatureMasking:  Randomly mask individual features (like BERT masking for LOB)
    5. TemporalJitter:  Slight random crop/shift within the time window
    6. SpreadPerturbation: Jitter the bid-ask spread while preserving mid-price
    7. LOBAugmentationPipeline: Composable pipeline of all the above

All augmentations operate on normalized tensors of shape (T, F) where
    T = sequence length (default 100 ticks)
    F = number of features (default 61)

Feature column layout (60-feature LOBSTER enriched set, interleaved per level):
    col 4*(i-1)+0: ask_price_i    col 4*(i-1)+1: ask_size_i
    col 4*(i-1)+2: bid_price_i    col 4*(i-1)+3: bid_size_i     (i = 1..10)
    cols 40-59:    microstructure features (micro_price, OFI, spread, etc.)
"""

import numpy as np
import torch
from typing import Optional, List


# ── Feature Column Index Ranges ────────────────────────────────────────────────
# Interleaved canonical layout: [ask_p, ask_s, bid_p, bid_s] per level
NUM_LEVELS      = 10
ASK_PRICE_COLS  = [4 * i + 0 for i in range(NUM_LEVELS)]  # ask_price_1 ... ask_price_10
ASK_VOL_COLS    = [4 * i + 1 for i in range(NUM_LEVELS)]  # ask_size_1  ... ask_size_10
BID_PRICE_COLS  = [4 * i + 2 for i in range(NUM_LEVELS)]  # bid_price_1 ... bid_price_10
BID_VOL_COLS    = [4 * i + 3 for i in range(NUM_LEVELS)]  # bid_size_1  ... bid_size_10
PRICE_COLS      = ASK_PRICE_COLS + BID_PRICE_COLS
VOLUME_COLS     = ASK_VOL_COLS + BID_VOL_COLS
MICRO_COLS      = list(range(40, 60))     # spread, imbalance, etc.


class PriceNoise:
    """
    Add small Gaussian noise to normalized price features.

    Prevents the model from memorizing specific price level patterns by
    perturbing prices slightly around their normalized values.

    Args:
        sigma: Standard deviation of noise relative to feature std (~0.01 recommended)
        price_cols: List of column indices corresponding to price features
    """
    def __init__(self, sigma: float = 0.01, price_cols: List[int] = PRICE_COLS):
        self.sigma = sigma
        self.price_cols = price_cols

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, F)
        noise = torch.randn(x.shape[0], len(self.price_cols)) * self.sigma
        x = x.clone()
        x[:, self.price_cols] += noise
        return x


class VolumeScaling:
    """
    Apply random multiplicative scaling to volume features.

    Volume distributions are heavy-tailed and highly variable across stocks
    and time periods. Random scaling teaches the model to rely on relative
    volume ratios (imbalance) rather than absolute sizes.

    Args:
        scale_range: (min, max) range for the random scaling factor
        vol_cols: List of column indices corresponding to volume features
    """
    def __init__(self, scale_range: tuple = (0.7, 1.3), vol_cols: List[int] = VOLUME_COLS):
        self.scale_range = scale_range
        self.vol_cols = vol_cols

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Single scale factor per sequence (realistic: volume bursts affect all levels)
        scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
        x = x.clone()
        x[:, self.vol_cols] *= scale
        return x


class LevelDropout:
    """
    Randomly zero out entire LOB depth levels.

    Simulates thin markets where deep levels are not quoted, or where data
    feeds occasionally miss quotes. Forces the model to predict robustly
    even when top-of-book levels are sufficient (not relying on deep levels).

    Args:
        p: Probability of dropping each depth level (per augmentation call)
        num_levels: Number of price levels in the LOB (default 10)
        min_levels_kept: Always keep at least this many levels (default 1 = best bid/ask)
    """
    def __init__(self, p: float = 0.15, num_levels: int = 10, min_levels_kept: int = 2):
        self.p = p
        self.num_levels = num_levels
        self.min_levels_kept = min_levels_kept

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        # Decide which levels to drop (never drop the best N levels)
        droppable_levels = list(range(self.min_levels_kept, self.num_levels))
        for level_idx in droppable_levels:
            if torch.rand(1).item() < self.p:
                # Zero out ask price, ask vol, bid price, bid vol for this level
                ask_p  = 4 * level_idx      # ask_price_{level}
                ask_v  = 4 * level_idx + 1  # ask_size_{level}
                bid_p  = 4 * level_idx + 2  # bid_price_{level}
                bid_v  = 4 * level_idx + 3  # bid_size_{level}
                x[:, [ask_p, ask_v, bid_p, bid_v]] = 0.0
        return x


class FeatureMasking:
    """
    Randomly mask individual features to zero.

    Inspired by BERT's masked language modeling, this forces the model to
    learn redundant representations — if one feature is masked, the model
    learns to infer it from correlated features (e.g., spread from bid/ask prices).

    Args:
        mask_prob: Probability of masking each feature per timestep
        protected_cols: Columns never masked (e.g., mid_price needed for backtest)
    """
    def __init__(self, mask_prob: float = 0.05, protected_cols: List[int] = []):
        self.mask_prob = mask_prob
        self.protected_cols = set(protected_cols)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # mask shape: (T, F), same mask applied across time for each feature
        T, F = x.shape
        mask = torch.rand(F) > self.mask_prob     # True = keep, False = mask
        for col in self.protected_cols:
            mask[col] = True                       # always keep protected cols
        x = x.clone()
        x[:, ~mask] = 0.0
        return x


class TemporalJitter:
    """
    Randomly shift the temporal window by a few ticks.

    Instead of always using exactly ticks [i, i+T], use ticks [i+j, i+j+T]
    where j is a small random offset. This is applied at the dataset level
    by cropping the sequence — prevents the model from learning position-
    specific patterns within the window.

    Note: Applied at dataset level, not directly here (see LOBSequenceDatasetAug).
    This class is a placeholder for documentation; actual implementation is in
    LOBSequenceDatasetAug via the jitter_ticks parameter.
    """
    pass


class SpreadPerturbation:
    """
    Jitter the bid-ask spread while approximately preserving mid-price.

    Widens or narrows the spread slightly by moving ask prices up and bid
    prices down (or vice versa) by a small random amount. Forces the model
    to be robust to varying spread conditions across different market regimes.

    Args:
        max_jitter: Maximum perturbation in normalized units
        ask_price_cols: Column indices for ask price levels
        bid_price_cols: Column indices for bid price levels
    """
    def __init__(
        self,
        max_jitter: float = 0.005,
        ask_price_cols: List[int] = ASK_PRICE_COLS,
        bid_price_cols: List[int] = BID_PRICE_COLS,
    ):
        self.max_jitter = max_jitter
        self.ask_price_cols = ask_price_cols
        self.bid_price_cols = bid_price_cols

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        jitter = torch.FloatTensor(1).uniform_(0, self.max_jitter).item()
        direction = 1 if torch.rand(1).item() > 0.5 else -1
        x = x.clone()
        x[:, self.ask_price_cols] += direction * jitter   # widen: ask goes up
        x[:, self.bid_price_cols] -= direction * jitter   # widen: bid goes down
        return x


class LOBAugmentationPipeline:
    """
    Composable LOB augmentation pipeline applied during training.

    Combines all augmentation strategies with configurable probabilities.
    Each augmentation is applied independently with its own activation
    probability, allowing flexible mixing.

    Args:
        price_noise_sigma:   Std dev for Gaussian price noise (0 = disabled)
        volume_scale_range:  (min, max) volume scaling factor (None = disabled)
        level_dropout_p:     Prob of dropping each deep LOB level (0 = disabled)
        feature_mask_prob:   Prob of masking each feature (0 = disabled)
        spread_jitter:       Max spread perturbation in normalized units (0 = disabled)
        apply_prob:          Probability of applying the ENTIRE pipeline per sample
                             (useful for training: apply aug to ~80% of samples)
    """
    def __init__(
        self,
        price_noise_sigma:  float = 0.01,
        volume_scale_range: Optional[tuple] = (0.8, 1.2),
        level_dropout_p:    float = 0.10,
        feature_mask_prob:  float = 0.03,
        spread_jitter:      float = 0.003,
        apply_prob:         float = 0.8,
    ):
        self.apply_prob = apply_prob
        self.transforms = []

        if price_noise_sigma > 0:
            self.transforms.append(PriceNoise(sigma=price_noise_sigma))

        if volume_scale_range is not None:
            self.transforms.append(VolumeScaling(scale_range=volume_scale_range))

        if level_dropout_p > 0:
            self.transforms.append(LevelDropout(p=level_dropout_p))

        if feature_mask_prob > 0:
            self.transforms.append(FeatureMasking(mask_prob=feature_mask_prob))

        if spread_jitter > 0:
            self.transforms.append(SpreadPerturbation(max_jitter=spread_jitter))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations with probability apply_prob."""
        if torch.rand(1).item() > self.apply_prob:
            return x    # Skip augmentation for this sample
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self) -> str:
        return (
            f"LOBAugmentationPipeline(\n"
            f"  transforms={[type(t).__name__ for t in self.transforms]},\n"
            f"  apply_prob={self.apply_prob}\n)"
        )


# ── Default augmentation presets ───────────────────────────────────────────────

def get_training_augmentation(strength: str = "medium") -> LOBAugmentationPipeline:
    """
    Get a preset augmentation pipeline for training.

    Args:
        strength: 'light', 'medium' (default), or 'heavy'

    Returns:
        LOBAugmentationPipeline configured for the chosen strength
    """
    presets = {
        "light": dict(
            price_noise_sigma=0.005,
            volume_scale_range=(0.9, 1.1),
            level_dropout_p=0.05,
            feature_mask_prob=0.02,
            spread_jitter=0.002,
            apply_prob=0.5,
        ),
        "medium": dict(
            price_noise_sigma=0.01,
            volume_scale_range=(0.8, 1.2),
            level_dropout_p=0.10,
            feature_mask_prob=0.03,
            spread_jitter=0.003,
            apply_prob=0.8,
        ),
        "heavy": dict(
            price_noise_sigma=0.02,
            volume_scale_range=(0.6, 1.4),
            level_dropout_p=0.20,
            feature_mask_prob=0.05,
            spread_jitter=0.005,
            apply_prob=0.9,
        ),
    }
    if strength not in presets:
        raise ValueError(f"Unknown strength '{strength}'. Choose 'light', 'medium', or 'heavy'.")
    return LOBAugmentationPipeline(**presets[strength])
