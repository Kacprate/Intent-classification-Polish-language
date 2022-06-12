import json

import numpy as np
import torch.utils.data as data_utils


class IntentDataset(data_utils.Dataset):
    def __init__(self, path, mode, random_seed) -> None:
        super().__init__()
        assert mode in ['train', 'val', 'test'], f"Dataset mode '{mode}' is not supported."
        np.random.seed(random_seed)
        
        extension = path.split('.')[-1]
        if extension == 'jsonl':
            json_objects = self.jsonl_load(path)
        else:
            raise NotImplementedError()
        
        self.__data = list()
        intents = set()
        for el in json_objects:
            utterance = el['utt']
            intent = el['intent']
            self.__data.append((utterance, intent))
            intents.add(intent)
        self.intents = sorted(intents)
        self.intent2ind = {self.intents[index]: index for index in range(len(self.intents))}
        self.ind2intent = {index: intent for intent, index in self.intent2ind.items()}
        np.random.shuffle(self.__data)
        
        if mode == 'train':
            data_min_idx = 0
            data_max_idx = int(len(self.__data) * 0.8) - 1
        elif mode == 'val':
            data_min_idx = int(len(self.__data) * 0.8)
            data_max_idx = int(len(self.__data) * 0.9) - 1
        else:
            data_min_idx = int(len(self.__data) * 0.9)
            data_max_idx = len(self.__data) - 1
            
        self.__data = self.__data[data_min_idx:data_max_idx]
        print(f'Loaded the MASSIVE ({path}) dataset with {len(self.__data)} utterances and {len(self.intents)} intents.')

    @staticmethod
    def jsonl_load(path):
        with open(path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        json_objects = [json.loads(line) for line in lines]
        return json_objects

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index):
        utterance, label = self.__data[index]
        return utterance, self.intent2ind[label]


if __name__ == '__main__':
    dataset = IntentDataset('./pl-PL.jsonl')
    print(dataset[2])
