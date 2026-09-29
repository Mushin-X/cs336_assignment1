from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
import os
from cs336_basics.tokenizer.tokenizer import Tokenizer
import time

#=====预分词=====

def pre_tokenization(chunk: str, special_tokens: list[str]) -> list[str]:
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    page = re.split(pattern, chunk)
    word = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for p in page:
        w = re.finditer(PAT, p)
        for i in w:
            word.append(i.group())

    return word

def pre_tokenization_parallel(
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    num_processes = 1,
) -> list[str]:
    with open(input_path, "rb") as f:
        bound = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        total_word = []

        # 后续改成并行，只要将下面的遍历改成同时分发给多个进程就行
        for start, end in zip(bound[:-1], bound[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore") 
            total_word.extend(pre_tokenization(chunk, special_tokens))

    return total_word

#=====合并=====

class WordInfo:
    def __init__(self, word: str):
        self.word = word
        self.count = 0
        word_b = word.encode("utf-8")
        self.word_state = [bytes([w]) for w in word_b]

    def merge_bp(self, bp: tuple[bytes, bytes]) -> None:
        new_state = []
        is_skip = 0
        for (b1, b2) in zip(self.word_state[:-1], self.word_state[1:]):
            if is_skip == 1:
                is_skip = 0
                continue
            if b1 != bp[0] or b2 != bp[1]:
                new_state.append(b1)
                continue

            new_state.append(b1 + b2)
            is_skip = 1

        if is_skip == 0 and len(self.word_state) > 0:
            new_state.append(self.word_state[-1])

        self.word_state = new_state.copy()

    def update_bpinfo(self, 
                 now_count_bp: dict[tuple[bytes, bytes], int], 
                 mulp: int = 1,
                 bp_to_word: dict[tuple[bytes, bytes], set[str]] = None,
                 is_add: int = 1
    ) -> set[tuple[bytes, bytes]]:
        #change_bp是此次字节对合并所引起改动的字节对（实际发生改动的可能要比存储的少）
        change_bp = set()
        for b1, b2 in zip(self.word_state[:-1], self.word_state[1:]):
            key = (b1, b2)

            change_bp.add(key)

            if key not in now_count_bp:
                now_count_bp[key] = 0
            now_count_bp[key] += mulp * self.count

            if bp_to_word != None:
                if is_add == 1:
                    if key not in bp_to_word:
                        bp_to_word[key] = set()
                    bp_to_word[key].add(self.word)
                else :
                    # 由于一个单词中可能有多个相同的字节对，导致出现这些字节对重复删同一个word的情况，因此不能用remove
                    bp_to_word[key].discard(self.word)

        return change_bp

#后续该部分可改成堆
def get_max(count: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    if len(count) == 0:
        raise ValueError("字节对字典为空，无法继续得到数量最多的字节对")

    (mb1, mb2), mx = max(count.items(), key=lambda k : (k[1], k[0])) #k[1]相同，则比较k[0]
    return (mb1, mb2)

def merge(
    word_info: dict[str, WordInfo], 
    count: dict[tuple[bytes, bytes], int], 
    bp_to_word: dict[tuple[bytes, bytes], set[str]],
    merge_num: int,
) -> list[tuple[bytes, bytes]]:
    merge_re = []
    start_time = time.perf_counter()
    for i in range(merge_num):
        if i % 100 == 0:
            ti = time.perf_counter() - start_time
            print(f"已经合并{i}次, 当前bpe训练时间已有 {ti} 秒")

        mx_pb = get_max(count)
        merge_re.append(mx_pb)

        s = bp_to_word[mx_pb].copy()
        change_bp = set()
        for word in s:
            # 更新字节对数量和状态，以及字节对所属单词的映射关系
            word_info[word].update_bpinfo(count, -1, bp_to_word, 0)
            word_info[word].merge_bp(mx_pb)
            chbp = word_info[word].update_bpinfo(count, 1, bp_to_word)

            # 后续若改成堆，只需将change_bp加入堆中，然后重新取mx
            # 但要注意取出的mx可能是旧值，此时应抛弃重新取，直到取到新值
            change_bp.update(chbp)
        
    return merge_re

def pre_count(total_word: list[str]) -> tuple[
    dict[str, WordInfo],
    dict[tuple[bytes, bytes], int], 
    dict[tuple[bytes, bytes], set[str]]
    ]:
    word_info = {}
    for word in total_word:
        if word not in word_info:
            word_info[word] = WordInfo(word)
        word_info[word].count += 1

    count = {}
    bp_to_word = {}
    for word, wi in word_info.items():
        wi.update_bpinfo(count, 1, bp_to_word)
    return word_info, count, bp_to_word

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    total_word = pre_tokenization_parallel(input_path, special_tokens)
    
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    id = 256
    for tok in special_tokens:
        vocab[id] = tok.encode("utf-8")
        id += 1

    word_info, count, bp_to_word = pre_count(total_word)
    merge_re = merge(word_info, count, bp_to_word, vocab_size - id)
    for x in merge_re:
        vocab[id] = x[0] + x[1]
        id += 1

    return vocab, merge_re