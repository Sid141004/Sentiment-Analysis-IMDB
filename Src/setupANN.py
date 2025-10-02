import torch
import torch.nn as nn  # <- needed for layers and Module

class FFNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, activation="relu", regularization=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Activation choice
        if activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "tanh":
            self.act = nn.Tanh()
        elif activation.lower() == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.5) if regularization=="dropout" else nn.Identity()

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = self.act(self.fc3(x))
        x = torch.sigmoid(self.out(x)).squeeze(-1)
        return x
