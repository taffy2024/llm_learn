# test.py
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# -----------------------------
# 加载数据集和模型
# -----------------------------
ds = load_dataset("ag_news")  # 下载 AG News 数据集
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置 padding token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # 确保 padding token 被考虑

# 冻结模型参数（只训练 soft prompt）
for param in model.parameters():
    param.requires_grad = False

# -----------------------------
# Prompt Tuning: 创建可训练 soft prompt
# -----------------------------
soft_prompt_length = 10
embedding_dim = model.transformer.wte.weight.shape[1]
soft_prompt = nn.Parameter(torch.randn(soft_prompt_length, embedding_dim))

class PromptTuningModel(nn.Module):
    def __init__(self, model, soft_prompt):
        super().__init__()
        self.model = model
        self.soft_prompt = soft_prompt

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        # 获取输入 embeddings
        input_embeds = self.model.transformer.wte(input_ids)
        # 扩展 soft prompt 到 batch
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        # 拼接 embeddings
        inputs_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        # 更新 attention_mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.soft_prompt.shape[0],
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # 更新 labels - soft prompt 部分不计算损失
        if labels is not None:
            prompt_labels = torch.full(
                (batch_size, self.soft_prompt.shape[0]),
                -100, device=labels.device, dtype=labels.dtype
            )
            labels = torch.cat([prompt_labels, labels], dim=1)

        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

prompt_model = PromptTuningModel(model, soft_prompt)

# -----------------------------
# 数据处理
# -----------------------------
def process_func(example):
    text = example["text"]
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # 因果语言模型，labels = input_ids
    return inputs

# tokenization
tokenized_ds = ds.map(process_func, remove_columns=ds['train'].column_names)

# -----------------------------
# TrainingArguments（改进版）
# -----------------------------
training_args = TrainingArguments(
    output_dir="./prompt_tuning_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=1e-3,           # 稳定训练
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    save_strategy="steps",         # 定期保存
    logging_dir="./logs",          # tensorboard 日志目录
    remove_unused_columns=False,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=prompt_model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
)

# -----------------------------
# 开始训练
# -----------------------------
trainer.train()

# -----------------------------
# 保存 soft prompt
# -----------------------------
torch.save(soft_prompt, "./prompt_tuning_output/soft_prompt.pt")
print("Training completed! Soft prompt saved.")
