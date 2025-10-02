# Step 1: Setup
import random
import numpy as np
import torch

# Reproducibility: set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Install HuggingFace datasets (if not already available in Colab)
!pip install datasets -q

# Load IMDB dataset
from datasets import load_dataset
imdb = load_dataset("imdb")

print(imdb)
