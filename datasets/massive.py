import json

import torch.utils.data as data_utils


class IntentDataset(data_utils.Dataset):
    def __init__(self, path, tokenizer) -> None:
        super().__init__()
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
            tokenized_utterance = tokenizer.encode(utterance, return_tensors="pt")
            self.__data.append((tokenized_utterance, intent))
            intents.add(intent)
        self.intents = sorted(intents)
        
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
        return self.__data[index]


if __name__ == '__main__':
    from transformers import HerbertTokenizer, RobertaModel

    tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
    dataset = IntentDataset('./pl-PL.jsonl', tokenizer)
    print(dataset[2])
