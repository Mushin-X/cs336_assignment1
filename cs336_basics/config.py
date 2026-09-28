from pathlib import Path

CONFIG = {
    "vocab_size": 512,
    "context_length": 64,
    "batch_size": 32,
    "d_model": 128,
    "num_layers": 2,
    "num_heads": 4,
    "d_ff": 512,
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

bpe_trainText_path = "tests/fixtures/tinystories_sample_5M.txt"
model_trainText_path = bpe_trainText_path

vocab_path = Path("data/vocab.json")
merges_path = Path("data/merges.txt")
model_save_path = Path("data/model_save.pt")