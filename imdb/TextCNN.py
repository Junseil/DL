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