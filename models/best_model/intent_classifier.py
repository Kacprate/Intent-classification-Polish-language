import torch
import torch.nn as nn


class IntentClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()
        self.__classification_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 96),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            
            nn.Dropout(p=0.15),            
            nn.Linear(96, output_dim),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x, hidden_state):
        x = torch.cat([x, hidden_state], dim=1)
        return self.__classification_head(x)
        