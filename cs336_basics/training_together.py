from cs336_basics.tokenizer import bpe_loading
from cs336_basics.train_data import data_loading, checkpointing
from cs336_basics.transformer import transformer_lm
from cs336_basics.train import adamw, cross_entropy, gradient_clipping, learning_rate_schedule
from cs336_basics.config import CONFIG, model_trainText_path, model_save_path

# ===== 加载/训练bpe =====
tk = bpe_loading.get_tokenizer()

# ===== 生成数据集生成器 =====
with open(model_trainText_path, "r") as f:
    text = f.read()
train_data = data_loading.DataLoading(
    tk.encode(text),
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

# ===== 训练 ======
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

    if step % 50 == 0 or step == CONFIG["total_steps"] - 1:
        print(f"step: {step}, loss: {loss}, lr: {lr}")

# 保存训练好的权重参数
checkpointing.save_checkpoint(model, optimizer, CONFIG["total_steps"], model_save_path)
