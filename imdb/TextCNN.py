import collections
import os

import nltk
nltk.download('punkt')
import torch
import tqdm.notebook as tqdm
from sklearn.model_selection import train_test_split
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torchtext.experimental.vocab_factory import Vocab

def load_dataset(dictionary, dev_ratio=None, using_vocab=None):
    print(f'loading files in {dictionary}')
    
    text = []
    labels = []
    classes = os.listdir(dictionary)
    
    for dictionary_name in classes:
        for filename in tqdm.tqdm(os.listdir(f'{dictionary}{dictionary_name}'), desc=f'loading {dictionary_name}'):
            with open(f'{dictionary}/{dictionary_name}/{filename}', encoding='utf-8') as f:
                tokens = tokenize(f.read(), max_length)
                text.append(tokens)
                labels.append(dictionary_name)
                
    if dev_ratio is not None:
        text, dev_text, labels, dev_labels = train_test_split(text, labels, test_size=dev_ratio)
        
    if using_vocab is None:
        using_vocab = make_vocab(text, vocab_size)
        
    text_transform = sequential_transforms(
        vocab_func(using_vocab),
        totenser(torch.long)
    )
    
    label_map = {name: index for index, name in enumerate(classes)}
    label_transform = sequential_transforms(
        lambda label: label_map[label],
        totensor(torch.long)
    )
    
    
    dataset = TextClassificationDataset(list(zip(labels, text)), using_vocab,
                                        (label_transform, text_transform))
    
    if dev_ratio is not None:
        dev_dataset = TextClassificationDataset(list(zip(dev_labels, dev_text)), using_vocab,
                                        (label_transform, text_transform))
        return dataset, dev_dataset
    else:
        return dataset

def tokenize(text, max_length):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = tokens[:max_length]
    tokens.extend(['<pad>']*(max_length-len(tokens)))
    return tokens

def make_vocab(text, vocab_size):
    counts = collections.Counter()
    for tokens in text:
        for token in tokens:
            counts[token] += 1
            
    _, max_count = counts.most_common(1)[0]
    counts['<pad>'] += max_count + 2
    counts['<unk>'] = max_count + 1
    vocab = Vocab(collections.OrderedDict(counts.most_common(vocab_size)))
    return vocab


class TextCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, filter_sizes, filter_counts, dropout_rate):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        layers = []

        for size, count in zip(filter_sizes, filter_counts):
            layers.append(nn.Conv1d(in_channels=embedding_dim, out_channels=count,
                                    kernel_size=size, padding=size-1))
            
        self.conv = nn.ModuleList(layers)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features=sum(filter_counts), out_features=num_classes)

    def forward(self, x):
        out = self.embedding(x) 
        out.transpose_(1, 2)

        features = []

        for layer in self.conv:
            feature = layer(out)
            feature, _ = feature.max(dim=2)
            features.append(feature)

        out = torch.cat(features, dim=1)
        out = self.relu(out)
        out = self.fc(out)

        return out

def evaluate(model, data_loader):
    with torch.no_grad():
        model.eval()
        num_corrects = 0
        num_total = 0
        for label, text in data_loader:
            label = label.to(device)
            text = text.to(device)

            output = model(text)
            predictions = output.argmax(dim=-1)
            num_corrects += (predictions == label).sum().item()
            num_total += text.size(0)

    return num_corrects/num_total