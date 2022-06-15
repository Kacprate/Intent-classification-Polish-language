import os
import shutil
import torch


def collate_fn_factory(tokenizer):
    def collate_fn(data):
        utterances, labels = zip(*data)
        tokenizer_output = tokenizer.batch_encode_plus(
            utterances,
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = torch.tensor(labels)
        return tokenizer_output, labels
    return collate_fn

def copy_models(models_dir, target_dir):
    model_files = os.listdir(models_dir)
    for model_file in model_files:
        path = os.path.join(models_dir, model_file)
        if os.path.isfile(path):
            target_path = os.path.join(target_dir, model_file)
            shutil.copyfile(path, target_path)