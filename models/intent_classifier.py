import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()
        self.__classification_head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        return self.__classification_head(x)
        