import torch
import torch.nn as nn
from torch.autograd import Variable


class IntentClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, device) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.__classification_head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            nn.Dropout(p=0.1),            
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1),
        )

        self.__lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.__lstm_postprocessing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, hidden_state, all_hidden_states):
        _, (h_n, _) = self.__lstm(all_hidden_states)
        h_n = h_n.squeeze(0)
        h_n = self.__lstm_postprocessing(h_n)
        x = torch.cat([x, h_n], dim=1)
        return self.__classification_head(x)
        