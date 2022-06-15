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
checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
if not os.path.isdir(checkpoints_dir):
    os.mkdir(checkpoints_dir)

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
    hidden_dim=768, output_dim=len(dataset.intents))
intent_classifier = intent_classifier.to(device)

optimizer = torch.optim.Adam(
    intent_classifier.parameters(), lr=config.learning_rate, betas=config.adam_betas)

loss_func = nn.BCELoss()

loss_list = []
val_loss_list = []
acc_list = []
val_acc_list = []

best_acc = 0.83

for epoch_index in range(1, config.epoch_count + 1):
    epoch_loss = 0
    val_epoch_loss = 0
    epoch_acc = 0
    val_epoch_acc = 0

    language_model.train()
    intent_classifier.train()
    loader_len = len(train_loader)
    for tokenizer_output, labels in tqdm(train_loader):
        tokenizer_output = {key: val.to(device)
                            for key, val in tokenizer_output.items()}
        labels_one_hot = nn.functional.one_hot(labels, len(dataset.intents))
        labels_one_hot = labels_one_hot.to(device).type(torch.float)

        with torch.no_grad():
            lm_outputs = language_model(**tokenizer_output)
        cls_hiddens = lm_outputs.pooler_output
        intents_pred = intent_classifier(cls_hiddens)

        loss = loss_func(intents_pred, labels_one_hot)
        epoch_loss += loss.item()

        intents_decoded = intents_pred.argmax(dim=1).cpu()
        accuracy = torch.sum(intents_decoded == labels).sum() / \
            intents_decoded.shape[0]
        epoch_acc += accuracy.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_acc /= len(train_loader)

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

            intents_decoded = intents_pred.argmax(dim=1).cpu()
            accuracy = torch.sum(
                intents_decoded == labels).sum() / intents_decoded.shape[0]
            val_epoch_acc += accuracy.item()

            loss = loss_func(intents_pred, labels_one_hot)
            val_epoch_loss += loss.item()

    val_epoch_acc /= len(test_loader)

    loss_list.append(epoch_loss)
    val_loss_list.append(val_epoch_loss)
    acc_list.append(epoch_acc)
    val_acc_list.append(val_epoch_acc)
    
    info_string = f'Epoch: {epoch_index}, train_loss: {epoch_loss:.4f}, val_loss: {val_epoch_loss:.4f}, epoch_acc: {epoch_acc:.4f}, val_acc: {val_epoch_acc:.4f}'
    print(info_string)
    
    with open(os.path.join(experiment_dir, 'log.txt'), mode='a') as f:
        f.write(f'{info_string}\n')
    
    if val_epoch_acc > best_acc:
        checkpoint_path = os.path.join(experiment_dir, f'best.pt')
        print(f'Saving model (better accuracy {best_acc:.4f} -> {val_epoch_acc:.4f}) {checkpoint_path}')
        torch.save(intent_classifier.state_dict(), checkpoint_path)
        with open(os.path.join(experiment_dir, 'best_info.txt'), mode='w') as f:
            f.write(info_string)
        best_acc = val_epoch_acc

    if epoch_index % config.save_every == 0:
        checkpoint_path = os.path.join(
            experiment_dir, 'checkpoints', f'checkpoint_intent_classifier_E_{epoch_index}_L_{val_epoch_acc:.4f}.pt')
        print(f'Saving checkpoint {checkpoint_path}')
        torch.save(intent_classifier.state_dict(), checkpoint_path)
        
    plt.plot(loss_list)
    plt.savefig(f'{experiment_dir}/loss.png')

    plt.cla()
    plt.plot(val_loss_list)
    plt.savefig(f'{experiment_dir}/val_loss.png')

    plt.cla()
    plt.plot(acc_list)
    plt.savefig(f'{experiment_dir}/acc.png')

    plt.cla()
    plt.plot(val_acc_list)
    plt.savefig(f'{experiment_dir}/val_acc.png')
