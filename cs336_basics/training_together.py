from cs336_basics.tokenizer import bpe_loading
from cs336_basics.train_data import data_loading, checkpointing
from cs336_basics.transformer import transformer_lm
from cs336_basics.train import adamw, cross_entropy, gradient_clipping, learning_rate_schedule
from cs336_basics.config import CONFIG
from cs336_basics.config import text_path, base_path, vocab_path, merges_path, best_model_save_path, latest_mode_save_path, config_path, log_path
from cs336_basics.config import final_best_model_save_path, final_config_path, final_latest_mode_save_path, final_log_path

import torch
import shutil
import time

base_path.mkdir(parents=True, exist_ok=True)

# ===== 划分数据集 =====
with open(text_path, "r") as f:
    text = f.read()
l = int(len(text) * 0.9)
train_text = text[:l]
val_text = text[l:]

# ===== 加载/训练bpe =====
tk = bpe_loading.get_tokenizer(
    train_text, 
    vocab_path=vocab_path, 
    merges_path=merges_path
)
# tk = bpe_loading.get_tokenizer() #若 model_data中已有BPE数据，可直接获取分词器

# ===== 生成数据集生成器 =====
train_data = data_loading.DataLoading(
    tk.encode(train_text),
    CONFIG["batch_size"],
    CONFIG["context_length"],
    CONFIG["device"],
)
val_data = data_loading.DataLoading(
    tk.encode(val_text),
    CONFIG["batch_size"],
    CONFIG["context_length"],
    CONFIG["device"],
)

# ===== 大模型 =====
model = transformer_lm.TransformerLm(
    CONFIG["vocab_size"],
    CONFIG["context_length"],
    CONFIG["d_model"],
    CONFIG["num_layers"],
    CONFIG["num_heads"],
    CONFIG["d_ff"],
    CONFIG["rope_theta"],
).to(CONFIG["device"])

# ===== 优化器 =====
optimizer = adamw.AdamW(
    model.parameters(),
    CONFIG["max_learning_rate"],
    CONFIG["betas"],
    CONFIG["eps"],
    CONFIG["weight_decay"],
)

# ===== 梯度裁剪 =====
clipper = gradient_clipping.GradientClipping(
    list(model.parameters()), CONFIG["max_l2_norm"]
)

# ===== 评估 =====
@torch.no_grad()
def evaluate(is_trainData=True):
    model.eval()
    sum = 0
    for _ in range(20):
        if is_trainData == True:
            data = train_data
        else:
            data = val_data

        inputs, targets = data.get_batch()
        out = model(inputs)
        out = out.view(-1, CONFIG["vocab_size"])
        targets = targets.view(-1)
        sum += cross_entropy.cross_entropy(out, targets)

    model.train()
    return sum / 20.0

# ===== 训练 ======
model.train()
mi_valLoss = 1e9
lossTr, lossVal, tiList, lrList = [], [], [], []
train_start_time = time.perf_counter()
shutil.copyfile("cs336_basics/config.py", config_path)
for step in range(CONFIG["total_steps"]):
    inputs, targets = train_data.get_batch()
    out = model(inputs)
    out = out.view(-1, CONFIG["vocab_size"])
    targets = targets.view(-1)
    loss = cross_entropy.cross_entropy(out, targets)

    optimizer.zero_grad()
    loss.backward()
    clipper.cal()

    lr = learning_rate_schedule.LearningRateSchedule(
        step,
        CONFIG["max_learning_rate"],
        CONFIG["min_learning_rate"],
        CONFIG["warmup_steps"],
        CONFIG["total_steps"],
    ).get_lr()
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.step()

    step += 1
    if step % 50 == 0 or step == CONFIG["total_steps"]:
        loss_tr = evaluate(is_trainData=True)
        loss_val = evaluate(is_trainData=False)
        ti = time.perf_counter() - train_start_time

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"step: {step:>6d} | "
                f"run_time: {ti:>10.4f} | "
                f"loss_train: {loss_tr:>8.6f} | "
                f"loss_val: {loss_val:>8.6f} | "
                f"lr: {lr:.7f}\n"
            )

        lossTr.append((step, loss_tr))
        lossVal.append((step, loss_val))
        tiList.append((step, ti))
        lrList.append((step, lr))
        print(f"step: {step:>6d} | run_time: {ti:>10.4f} | loss_train: {loss_tr:>8.6f} | loss_val: {loss_val:>8.6f} | lr: {lr:>8.6f} | ti: {ti:>6.4f}")

    if step % 500 == 0:
        # ===== 保存检查点 =====
        checkpointing.save_checkpoint(model, optimizer, step, latest_mode_save_path)
        # 保存在验证集上评估最好时的权重参数
        if loss_val < mi_valLoss:
            checkpointing.save_checkpoint(model, optimizer, step, best_model_save_path)
            mi_valLoss = loss_val

shutil.copyfile(best_model_save_path, final_best_model_save_path)
shutil.copyfile(latest_mode_save_path, final_latest_mode_save_path)
shutil.copyfile(config_path, final_config_path)
shutil.copyfile(log_path, final_log_path)
# ===== 训练过程图（补充） =====
