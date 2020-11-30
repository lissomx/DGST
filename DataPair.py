import re
import os
import random
import hashlib
import pickle
import json
from itertools import groupby
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class DataPair(Dataset):
    def __init__(self, data0_path, data1_path, min_word_count=4, base_corpus=None, model_path="./model_save/", amount=1, device=None):
        with open(data0_path) as f:
            data0 = [s.strip().lower() for s in f.readlines()]
        with open(data1_path) as f:
            data1 = [s.strip().lower() for s in f.readlines()]
        corpus = " ".join(data0 + data1)
        xcode = hashlib.sha1(f"{corpus}-{min_word_count}-{device}-{base_corpus.xcode if base_corpus is not None else 0}".encode('utf-8'))
        self.xcode = int(xcode.hexdigest(), 16) % 10**8
        model_file_path = f"{model_path}DataPair_{self.xcode}.pk"
        if os.path.exists(model_file_path):
            info = torch.load(model_file_path)
            print(model_file_path)
            self.xcode = info["xcode"]
            self.data0 = info["data0"]
            self.data1 = info["data1"]
            self.word_id = info["word_id"]
            self.id_word = info["id_word"]
        else:
            self._make_dic(corpus, min_word_count, base_corpus)
            label0 = torch.tensor([1.0,-1.0], device=device)
            label1 = torch.tensor([-1.0,1.0], device=device)
            self.data0 = [self.sentence_to_tensor(s.split(" "), device=device) for s in data0]
            self.data1 = [self.sentence_to_tensor(s.split(" "), device=device) for s in data1]
            info = {}
            info["xcode"] = self.xcode 
            info["data0"] = self.data0
            info["data1"] = self.data1
            info["word_id"] = self.word_id
            info["id_word"] = self.id_word
            torch.save(info, model_file_path)
        self.data0 = info["data0"][:int(len(self.data0)*amount)]
        self.data1 = info["data1"][:int(len(self.data1)*amount)]
        self.vocab_size = len(self.word_id)
        self.data0_len = len(self.data0)
        self.data1_len = len(self.data1)
    
    def _make_dic(self, corpus, min_word_count, base_corpus=None):
        if base_corpus is not None:
            self.word_id = base_corpus.word_id
            self.id_word = base_corpus.id_word
        else:
            corpus = corpus.split(" ")
            words = sorted(corpus)
            group = groupby(words)
            word_count = [(w, sum(1 for _ in c)) for w, c in group]
            word_count = [(w, c) for w, c in word_count if c >= min_word_count]
            word_count.sort(key=lambda x: x[1], reverse=True)
            word_id = dict([(w, i+4) for i, (w, _) in enumerate(word_count)])
            word_id["<pad>"] = 0
            word_id["<unk>"] = 1
            word_id["<sos>"] = 2
            word_id["<eos>"] = 3
            self.word_id = word_id
            self.id_word = dict([(i, w) for w, i in word_id.items()])
    
    def sentence_to_tensor(self, sentence, device):
        v = [self.word_id.get(w, 1) for w in sentence]
        v = [2]+v+[3]
        v = torch.tensor(v, device=device)
        return v
    
    def shuffle(self):
        random.shuffle(self.data0)
        random.shuffle(self.data1)

    def __getitem__(self, index):
        index0 = index1 = index
        b_size = index.stop-index.start
        if index0.stop > self.data0_len:
            s = random.randint(0,self.data0_len-b_size-1)
            index0 = slice(s, s+b_size)
        if index1.stop > self.data1_len:
            s = random.randint(0,self.data1_len-b_size-1)
            index1 = slice(s, s+b_size)
        return (self.data0[index0], self.data1[index1])

    def __len__(self):
        length = max(self.data0_len,self.data1_len)
        return length
    
    def totext(self, sen):
        text = [self.id_word[i] for i in sen]
        return " ".join(text)
        
    
        