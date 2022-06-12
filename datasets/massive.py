import torch.utils.data as data_utils

class IntentDataset(data_utils.Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __getitem__(self):
        raise NotImplementedError()