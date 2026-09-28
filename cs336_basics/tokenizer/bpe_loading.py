import json
from cs336_basics.tokenizer import train_bpe, tokenizer
from cs336_basics.config import CONFIG, special_tokens, bpe_trainText_path, vocab_path, merges_path

from pathlib import Path

def load_bpe(vocab_path, merges_path):
    with open(vocab_path) as f:
        vocab = {int(i):bytes.fromhex(b) for i, b in json.load(f).items()}

    with open(merges_path) as f:
        merges = [tuple(bytes.fromhex(x) for x in line.split()) for line in f]

    return vocab, merges

def save_bpe(vocab, vocab_path, merges, merges_path):
    with open(vocab_path, "w") as f:
        json.dump({str(i):b.hex() for i, b in vocab.items()}, f)

    with open(merges_path, "w") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")

def get_tokenizer():
    if vocab_path.exists() and merges_path.exists():
        vocab, merges = load_bpe(vocab_path, merges_path)
    else:
        vocab, merges = train_bpe.run_train_bpe(
            bpe_trainText_path, CONFIG["vocab_size"], special_tokens
        )
        save_bpe(vocab, vocab_path, merges, merges_path)

    return tokenizer.Tokenizer(vocab, merges, special_tokens)