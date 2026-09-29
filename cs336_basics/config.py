from pathlib import Path

CONFIG = {
    "vocab_size": 1024,
    "context_length": 128,
    "batch_size": 32,
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": 1024,
    "rope_theta": 10000,
    "device": "mps",
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.1,
    "total_steps": 10_000,
    "max_learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 500,
    "max_l2_norm": 1.0,
}

special_tokens = ["<|endoftext|>"]

text_path = "data/TinyStoriesV2-GPT4-valid.txt"

vocab_path = Path("data/vocab.json")
merges_path = Path("data/merges.txt")
best_model_save_path = Path("data/best_model_save.pt")
latest_mode_save_path = Path("data/latest_model_save.pt")