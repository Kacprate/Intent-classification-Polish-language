import argparse
import importlib
import logging
import os
import types
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from tqdm import tqdm
from transformers import HerbertTokenizer, RobertaModel

import utils
from datasets.massive import IntentDataset


@dataclass
class ModelEvaluator():

    logger: logging.Logger
    language_model: nn.Module
    intent_classifier: nn.Module
    intents_len: int
    test_loader: data_utils.DataLoader
    device: torch.device

    def evaluate(self):
        self.language_model.eval()
        self.intent_classifier.eval()

        intents_list = []
        labels_list = []
        val_epoch_acc = 0
        with torch.no_grad():
            for tokenizer_output, labels in tqdm(self.test_loader):
                tokenizer_output = {key: val.to(self.device)
                                    for key, val in tokenizer_output.items()}
                labels_one_hot = nn.functional.one_hot(
                    labels, self.intents_len)
                labels_one_hot = labels_one_hot.to(
                    self.device).type(torch.float)

                lm_outputs = self.language_model(**tokenizer_output)
                cls_hiddens = lm_outputs.pooler_output
                hidden_state = lm_outputs.last_hidden_state.mean(dim=1)
                intents_pred = self.intent_classifier(
                    cls_hiddens, hidden_state)

                intents_decoded = intents_pred.argmax(dim=1).cpu()
                accuracy = torch.sum(
                    intents_decoded == labels).sum() / intents_decoded.shape[0]
                val_epoch_acc += accuracy.item()

                # [0] because batch size is 1
                intents_to_save = intents_pred[0].cpu().numpy()
                label_to_save = labels[0]

                intents_list.append(intents_to_save)
                labels_list.append(label_to_save)
                
        accuracy = val_epoch_acc / len(self.test_loader)
        self.logger.info(f'Evaluation accuracy: {accuracy}')

        labels_np = np.array(labels_list)
        intents_np = np.array(intents_list)

        return intents_np, labels_np


def load_exp_modules(exp_path: str) -> Tuple[types.ModuleType]:
    config_path = os.path.join(exp_path, 'config')
    config_path_str = str(config_path).replace('/', '.')
    config = importlib.import_module(config_path_str)
    model_path = os.path.join(exp_path, 'models', 'intent_classifier')
    model_path_str = str(model_path).replace('\\', '/').replace('/', '.')
    model = importlib.import_module(model_path_str)
    return config, model


def main():
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None,
                        help="Experiment folder path.")
    args = parser.parse_args()
    
    exp_path = os.path.abspath(args.exp)

    config_module, IntentClassifier_module = load_exp_modules(args.exp)
    config = config_module.Config()
    IntentClassifier = IntentClassifier_module.IntentClassifier

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = HerbertTokenizer.from_pretrained(
        "allegro/herbert-klej-cased-tokenizer-v1")

    collate_fn = utils.collate_fn_factory(tokenizer)

    test_dataset = IntentDataset(
        path=config.dataset_path, mode='val', random_seed=config.dataset_random_seed)
    test_loader = data_utils.DataLoader(test_dataset,
                                        batch_size=1,  # 1 for testing purposes
                                        shuffle=True,
                                        collate_fn=collate_fn)

    language_model = RobertaModel.from_pretrained(
        "allegro/herbert-klej-cased-v1", is_decoder=False)
    language_model = language_model.to(device)

    intent_classifier = IntentClassifier(
        hidden_dim=768, output_dim=len(test_dataset.intents))
    model_path = os.path.join(exp_path, 'best.pt')
    intent_classifier.load_state_dict(torch.load(model_path))
    intent_classifier = intent_classifier.to(device)

    intents_len = len(test_dataset.intents)
    model_evaluator = ModelEvaluator(
        logger, language_model, intent_classifier, intents_len, test_loader, device)
    intents_np, labels_np = model_evaluator.evaluate()

    intents_df = pd.DataFrame(intents_np)
    labels_df = pd.DataFrame({60: labels_np})
    df = intents_df.join(labels_df)
    
    save_path = os.path.join(exp_path, 'test_results.csv')
    df.to_csv(save_path, index=False, header=False)
    logger.info(f'A .csv file with intent preditcion vectors and labels saved to {save_path}.')


if __name__ == "__main__":
    main()
