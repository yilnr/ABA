"""Minimal defaults shared by the KS experiment configuration."""

config = {
    "dataset": {},
    "visual": {"hidden_dim": 512},
    "train": {"local_rank": 0},
    "head": {"type": "MLP"},
    "output_dir": "",
    "debug": False,
}
