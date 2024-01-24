import os
import pandas as pd
import torch
import re
from nltk.corpus import stopwords 
from collections import Counter
import numpy as np


def train_loop(train_loader, model, loss_fn, optimizer, device):
    model.train()

    size = len(train_loader.dataset)
    num_batches = len(train_loader)

    train_loss, train_correct = 0, 0

    for word_embed, labels in train_loader:
        # Transfering images and labels to GPU if available
        word_embed, labels = word_embed.to(device), labels.to(device)
        
        # Forward pass 
        outputs = model(word_embed)
        outputs = outputs.type(torch.float64)

        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predicted = torch.round(outputs)
        
        train_correct += (predicted == labels).type(torch.float).sum().item()


    train_loss /= num_batches
    train_correct /=size
    
    return train_loss, train_correct

def test_loop(test_loader, model, loss_fn, device):
    model.eval()

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, test_correct = 0, 0

    with torch.no_grad():
        for word_embed, labels in test_loader:

            word_embed, labels = word_embed.to(device), labels.to(device)

            outputs = model(word_embed)
            outputs = outputs.type(torch.float64)

            test_loss += loss_fn(outputs, labels).item()

            predicted = torch.round(outputs)
            test_correct += (predicted == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    test_correct /= size
    
    return test_loss, test_correct


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train,x_val, length = 2000, stop_word_active = True):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)

            if stop_word_active:
                 if word not in stop_words and word != '':
                    word_list.append(word)
            else:
                if word != '':
                    word_list.append(word)
  
    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:length]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
        
    print(len(final_list_train))
    return final_list_train, final_list_test,onehot_dict

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

