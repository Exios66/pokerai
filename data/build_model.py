import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset

tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")
vocab_size = len(tokenizer)

# ---- Model: random init (from_config, not from_pretrained) ----
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=192,
    n_embd=256,
    n_layer=6,
    n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Training on:", device)

# ---- Data: tokenize full sequence, mask everything up to the last comma ----
raw = load_dataset("text", data_files={"train": "data/hands_fold_update.txt"})["train"]
raw = raw.train_test_split(test_size=0.05, seed=42)

def encode_with_mask(text):
    last_comma = text.rfind(",")
    prompt_text = text[:last_comma + 1]  # context (state), up to and including last comma

    prompt_ids = tokenizer(prompt_text)["input_ids"]
    full_ids = [tokenizer.bos_token_id] + tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
    prompt_len = len(prompt_ids) + 1  # +1 for BOS

    labels = [-100] * len(full_ids)
    for i in range(prompt_len, len(full_ids)):
        labels[i] = full_ids[i]
    return full_ids, labels

class HandsDataset(Dataset):
    def __init__(self, split):
        self.examples = [encode_with_mask(ex["text"]) for ex in split]
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def collate(batch):
    max_len = max(len(ids) for ids, _ in batch)
    pad_id = tokenizer.pad_token_id
    input_ids, labels, attn_mask = [], [], []
    for ids, lbls in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        labels.append(lbls + [-100] * pad_len)
        attn_mask.append([1] * len(ids) + [0] * pad_len)
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn_mask)

train_ds = HandsDataset(raw["train"])
val_ds = HandsDataset(raw["test"])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate)

# ---- Sanity-check the masking on one real example ----
ids, labels = train_ds[0]
print("\nTokens: ", tokenizer.convert_ids_to_tokens(ids))
print("Labels: ", [tokenizer.convert_ids_to_tokens([t])[0] if t != -100 else "---" for t in labels])
print("(Only non-'---' positions contribute to the loss -- verify that's just the decision.)\n")

# ---- Raw training loop ----
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def run_eval():
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for input_ids, labels, attn_mask in val_loader:
            input_ids, labels, attn_mask = input_ids.to(device), labels.to(device), attn_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100)
            n_tok = (targets != -100).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
    model.train()
    return total_loss / total_tokens

step = 0
for epoch in range(3):
    for input_ids, labels, attn_mask in train_loader:
        input_ids, labels, attn_mask = input_ids.to(device), labels.to(device), attn_mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"epoch {epoch} step {step:5d} | train loss {loss.item():.4f}")
        step += 1

    print(f"== end of epoch {epoch}: val loss {run_eval():.4f} ==")

print("\nBigram baseline val loss was ~1.53 -- compare final val loss above to that.")


