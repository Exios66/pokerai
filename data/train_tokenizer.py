import os
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

# Learn a vocab from your hand histories.
# vocab_size is small on purpose -- your domain vocabulary is tiny
# compared to English, so we don't need GPT-2's 50k tokens.
tokenizer.train(
    files=["data/hands.txt"],
    vocab_size=4000,
    min_frequency=2,
    special_tokens=["<|startoftext|>", "<|pad|>"],
)

# Create tokenizer directory if it doesn't exist
os.makedirs("tokenizer", exist_ok=True)
tokenizer.save("tokenizer/tokenizer.json")  # saves in the correct JSON format

# Wrap it in a Transformers-compatible tokenizer class so it works with
# AutoModel / TRL later.
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
hf_tokenizer.add_special_tokens({
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "eos_token": "<|startoftext|>",
})
hf_tokenizer.save_pretrained("tokenizer")

# Sanity check: tokenize one real hand and print it.
sample = open("data/hands.txt", encoding="utf-8").readline().strip()
ids = hf_tokenizer(sample)["input_ids"]
print("Sample hand:", sample)
print("Token count:", len(ids))

# Check the length distribution across all hands -- this tells us
# what context length (max_position_embeddings) the model needs.
lines = open("data/hands.txt", encoding="utf-8").readlines()
lengths = [len(hf_tokenizer(l.strip())["input_ids"]) for l in lines]
print("Max tokens in a hand:", max(lengths))
print("Avg tokens per hand:", sum(lengths) / len(lengths))
