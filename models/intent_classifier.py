import torch.nn as nn
from transformers import RobertaModel

class IntentClassifier(nn.Module):
    def __init__(self, hiddem_dim, output_dim) -> None:
        super().__init__()
        self.__classification_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hiddem_dim, 256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.1),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        return self.__classification_head(x)
        
        