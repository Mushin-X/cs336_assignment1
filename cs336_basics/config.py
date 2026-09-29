from pathlib import Path
from datetime import datetime

CONFIG = {
    "vocab_size": 512,
    "context_length": 64,
    "batch_size": 16,
    "d_model": 128,
    "num_layers": 2,
    "num_heads": 4,
    "d_ff": 512,
    "rope_theta": 10000,
    "device": "mps",
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.1,
    "total_steps": 5000,
    "max_learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 100,
    "max_l2_norm": 1.0,
}

special_tokens = ["<|endoftext|>"]

text_path = "data/TinyStoriesV2-GPT4-valid.txt"

now_time = datetime.now().strftime("%Y%m%d_%H%M%S")
base_path = Path("experiments") / now_time

vocab_path = base_path / "vocab.json"
merges_path = base_path / "merges.txt"
best_model_save_path = base_path / "best_model_save.pt"
latest_mode_save_path = base_path / "latest_model_save.pt"
config_path = base_path / "config.py"
log_path = base_path / "log.txt"