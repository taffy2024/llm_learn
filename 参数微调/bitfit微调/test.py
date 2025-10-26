# bitfit_training_compatible.py
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# -----------------------------
# 1. 加载数据集
# -----------------------------
ds = load_dataset("lyuricky/alpaca_data_zh_51k", split="train")
ds = ds.shuffle(seed=42).select(range(100))  # 小样本调试

# -----------------------------
# 2. 加载 tokenizer
# -----------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 自动设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# 3. 数据预处理函数
# -----------------------------
def process_func(examples):
    instructions = [str(ins) for ins in examples["instruction"]]
    inputs_field = [str(inp) if inp is not None else "" for inp in examples["input"]]
    targets = [str(tar) for tar in examples["output"]]

    prompts = [
        (ins + ("\n" + inp if inp.strip() else "")).strip()
        for ins, inp in zip(instructions, inputs_field)
    ]
    full_texts = [(prompt + "\n" + tar).strip() for prompt, tar in zip(prompts, targets)]

    tokenized_full = tokenizer(
        full_texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors=None
    )

    input_ids, labels = [], []

    for i, full_text in enumerate(full_texts):
        prompt_tokens = tokenizer(prompts[i], add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        full_tokens = tokenized_full["input_ids"][i]

        label = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]
        # 截断或填充到 max_length
        if len(label) < 512:
            label += [-100] * (512 - len(label))
        elif len(label) > 512:
            label = label[:512]

        input_ids.append(full_tokens)
        labels.append(label)

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels
    }

tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)

split_ds = tokenized_ds.train_test_split(test_size=0.2, seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]

# -----------------------------
# 4. 加载模型
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只解冻 bias 参数（BitFit）
for name, param in model.named_parameters():
    if "bias" in name:
        param.requires_grad = True

# 可训练参数统计
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

# -----------------------------
# 5. Data Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------------
# 6. 训练参数（兼容旧版本 transformers）
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bitfit_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-3,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    warmup_ratio=0.1
)

# -----------------------------
# 7. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# -----------------------------
# 8. 开始训练
# -----------------------------
trainer.train()

# 评估
results = trainer.evaluate()
print("评估结果（loss）:", results)

# 保存模型
trainer.save_model("./bitfit_model")
print("模型已保存到 ./bitfit_model")
