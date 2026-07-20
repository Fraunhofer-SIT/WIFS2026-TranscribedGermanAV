import torch
from torch import nn

h1_dim = 256
h2_dim = 128
h3_dim = 64
dropout_p = 0.01


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(embedding_dim, h1_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h1_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h2_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(h2_dim, h3_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h3_dim),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, txt1, txt2):
        out1 = self.nn(txt1)
        out2 = self.nn(txt2)
        return torch.nn.functional.cosine_similarity(out1, out2)
