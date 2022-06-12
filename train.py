import os
import shutil

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import HerbertTokenizer, RobertaModel

from config import Config
from datasets.massive import IntentDataset
from models.intent_classifier import IntentClassifier


config = Config()
experiment_dir = os.path.abspath(f'./results/{config.experiment_name}')
if not os.path.isdir(experiment_dir):
    os.mkdir(experiment_dir)
shutil.copyfile('./config.py', f'{experiment_dir}/config.py')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = HerbertTokenizer.from_pretrained(
    "allegro/herbert-klej-cased-tokenizer-v1")


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


dataset = IntentDataset(path=config.dataset_path,
                        mode='train', random_seed=config.dataset_random_seed)
train_loader = data_utils.DataLoader(dataset,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     collate_fn=collate_fn)

val_dataset = IntentDataset(
    path=config.dataset_path, mode='val', random_seed=config.dataset_random_seed)
test_loader = data_utils.DataLoader(val_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    collate_fn=collate_fn)

language_model = RobertaModel.from_pretrained(
    "allegro/herbert-klej-cased-v1", is_decoder=False)
language_model = language_model.to(device)

intent_classifier = IntentClassifier(
    hiddem_dim=768, output_dim=len(dataset.intents))
intent_classifier = intent_classifier.to(device)

optimizer = torch.optim.Adam(
    intent_classifier.parameters(), lr=config.learning_rate, betas=config.adam_betas)

loss_func = nn.BCELoss()

loss_list = []
val_loss_list = []

for epoch_index in range(config.epoch_count):
    epoch_loss = 0
    val_epoch_loss = 0

    language_model.train()
    intent_classifier.train()
    loader_len = len(train_loader)
    for tokenizer_output, labels in tqdm(train_loader):
        tokenizer_output = {key: val.to(device)
                            for key, val in tokenizer_output.items()}
        labels_one_hot = nn.functional.one_hot(labels, len(dataset.intents))
        labels_one_hot = labels_one_hot.to(device).type(torch.float)

        lm_outputs = language_model(**tokenizer_output)
        cls_hiddens = lm_outputs.pooler_output
        intents_pred = intent_classifier(cls_hiddens)

        loss = loss_func(intents_pred, labels_one_hot)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    language_model.eval()
    intent_classifier.eval()
    with torch.no_grad():
        for tokenizer_output, labels in tqdm(test_loader):
            tokenizer_output = {key: val.to(device)
                                for key, val in tokenizer_output.items()}
            labels_one_hot = nn.functional.one_hot(
                labels, len(dataset.intents))
            labels_one_hot = labels_one_hot.to(device).type(torch.float)

            lm_outputs = language_model(**tokenizer_output)
            cls_hiddens = lm_outputs.pooler_output
            intents_pred = intent_classifier(cls_hiddens)

            loss = loss_func(intents_pred, labels_one_hot)
            val_epoch_loss += loss.item()

    loss_list.append(epoch_loss)
    val_loss_list.append(val_epoch_loss)

    print(
        f'Epoch: {epoch_index}, train_loss: {epoch_loss:.4f}, val_loss: {val_epoch_loss:.4f}')


torch.save(intent_classifier.state_dict(),
           f'{experiment_dir}/intent_classifier.pt')

plt.plot(loss_list)
plt.savefig(f'{experiment_dir}/loss.png')

plt.cla()
plt.plot(val_loss_list)
plt.savefig(f'{experiment_dir}/val_loss.png')
