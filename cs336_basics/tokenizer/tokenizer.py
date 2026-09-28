from typing import Iterable, Iterator

import regex as re

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merges_to_id = {bp : i for i, bp in enumerate(self.merges)}
        self.btoi = {b : i for i, b in self.vocab.items()}

        if self.special_tokens is not None and len(self.special_tokens) != 0:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)

    def pre_tokenization(self, chunk: str) -> list[str]:
        if self.special_tokens is not None and len(self.special_tokens) != 0:
            #和train_bpe不同，这里应保留特殊字符
            page = re.split(f"({self.pattern})", chunk)
        else:
            page = [chunk]
        word = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for p in page:
            if self.special_tokens != None and p in self.special_tokens:
                word.append(p)
                continue

            w = re.finditer(PAT, p)
            for i in w:
                word.append(i.group())

        return word

    #每次合并只关注一种字节对，因为每合并完一次后需要重新去判断在合并规则中最靠前的字节对
    def merge_bp(
        self, 
        ids: list[bytes], 
        bp: tuple[bytes, bytes]
    ) -> list[bytes]:
        new_ids = []
        is_skip = 0
        for (a, b) in zip(ids[:-1], ids[1:]):
            if is_skip == 1:
                is_skip = 0
                continue
            if a != bp[0] or b != bp[1]:
                new_ids.append(a)
                continue

            new_ids.append(a + b)
            is_skip = 1 
        if is_skip == 0:
            new_ids.append(ids[-1])
        return new_ids

    #由于不能跨单词合并，因此每个单词独立编码
    def encode_word(self, word: str) -> list[int]:
        ids = [bytes([i]) for i in word.encode("utf-8")]
        while True:
            now_merge = []
            for (a, b) in zip(ids[:-1], ids[1:]):
                if (a, b) in self.merges_to_id:
                    now_merge.append((self.merges_to_id[(a,b)], (a, b)))

            if len(now_merge) == 0:
                break

            ids = self.merge_bp(ids, min(now_merge)[1])

        return [self.btoi[b] for b in ids]

    def encode(self, text: str) -> list[int]:
        total_word = self.pre_tokenization(text)
        ids = []
        for word in total_word:
            if self.special_tokens != None and word in self.special_tokens:
                ids.append(self.btoi[word.encode("utf-8")])
            else:
                ids.extend(self.encode_word(word))

        return ids

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[i] for i in ids)
        return text.decode("utf-8", errors="replace")    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def is_end(self, token_id: int) -> bool:
        if self.special_tokens is None:
            return False

        special_tokens_set = {
            tok.encode("utf-8") for tok in self.special_tokens
        }
        # 不能用self.vocab[token_id].decode("utf-8")，因为vocab的值可能是单个字节，而不是字符串编码过来的utf-8
        if self.vocab[token_id] in special_tokens_set:
            return True
        return False