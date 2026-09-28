from cs336_basics.transformer import transformer_lm, softmax
from cs336_basics.train_data import checkpointing, data_loading
from cs336_basics.tokenizer import bpe_loading
from cs336_basics.config import CONFIG, model_save_path
import torch

@torch.no_grad()
def generate_next_token(input: list[int], temperatrue, top_p) -> int:
    ids = torch.tensor(input[-CONFIG["context_length"]:], dtype=torch.long, device=CONFIG["device"])
    ids = ids.view(1, -1) #(1, seq_len)
    logits = model(ids)[0, -1, :] #(vocab_size)
    p = softmax.softmax(logits / temperatrue, dim=0)
    sort_p, sort_id = torch.sort(p, descending=True)
    l = len(p)
    sum, pos = 0, 0
    for i in range(l):
        sum += sort_p[i]
        if sum >= top_p:
            pos = i
            break

    sort_p = sort_p[:pos+1]
    sort_p = sort_p / sort_p.sum()
    sort_id = sort_id[:pos+1]
    return sort_id[torch.multinomial(sort_p, num_samples=1).item()].item() #不加item()会返回一个长度为 1 的torch.long，加了后变成 int

def generate_text(input: str, max_len, temperatrue, top_p) -> str:
    l = len(input)
    if max_len < 0:
        raise ValueError(f"设定的生成文本长度(max_len={max_len})为负值")
    if max_len == 0:
        return input

    tk = bpe_loading.get_tokenizer()
    ids = tk.encode(input)

    for i in range(max_len):
        new_id = generate_next_token(ids, temperatrue, top_p)
        if tk.is_end(new_id):
            break

        ids.append(new_id)

    return tk.decode(ids)

model = transformer_lm.TransformerLm(
    CONFIG["vocab_size"],
    CONFIG["context_length"],
    CONFIG["d_model"],
    CONFIG["num_layers"],
    CONFIG["num_heads"],
    CONFIG["d_ff"],
    CONFIG["rope_theta"],
).to(CONFIG["device"])
checkpointing.load_checkpoint(model_save_path, model)

input_text = "Hello, the weather looks very nice today, I plan to"
print(generate_text(input_text, max_len=500, temperatrue=0.8, top_p=0.9))




