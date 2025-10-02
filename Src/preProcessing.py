from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# 1. Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 200
BATCH_SIZE = 32

# 2. Custom PyTorch dataset
class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),       # [seq_len]
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# 3. Create dataset objects
train_dataset = IMDBDataset(imdb["train"], tokenizer, MAX_LEN)
test_dataset = IMDBDataset(imdb["test"], tokenizer, MAX_LEN)

# 4. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"Step 2 done: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
