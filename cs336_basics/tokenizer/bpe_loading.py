import json
from cs336_basics.tokenizer import train_bpe, tokenizer
from cs336_basics.config import CONFIG, special_tokens

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

def get_tokenizer(
    text=None, 
    is_retrain=False, 
    vocab_path=None, 
    merges_path=None
):
    if is_retrain == False and vocab_path.exists() and merges_path.exists():
        print("直接加载已训好的 BPE 数据")
        vocab, merges = load_bpe(vocab_path, merges_path)
    else:
        if text is None:
            raise ValueError("没有传入BPE训练文本, 无法训练")

        bpe_trainText_path = Path("data/bpe_train.txt")
        with open(bpe_trainText_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("开始训练 BPE ")
        vocab, merges = train_bpe.run_train_bpe(
            bpe_trainText_path, CONFIG["vocab_size"], special_tokens
        )
        bpe_trainText_path.unlink()

        save_bpe(vocab, vocab_path, merges, merges_path)
        print(f"已完成 BPE 训练, 合并了{len(merges)}次")

    return tokenizer.Tokenizer(vocab, merges, special_tokens)