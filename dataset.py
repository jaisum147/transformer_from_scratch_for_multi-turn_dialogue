import pandas as pd
import torch
from torch.utils.data import Dataset
import nltk
from nltk.tokenize import word_tokenize

class DialogueDataset(Dataset):
    def __init__(self, csv_file, vocab=None, max_len=50):
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len

        self.texts = []
        self.labels = []

        for _, row in self.data.iterrows():
            # dialog is stored correctly as a Python-style list
            dialog = row["dialog"].strip("[]").split("','")
            dialog = [d.replace("'", "").strip() for d in dialog]

            # emotion is space-separated numbers → convert safely
            emotion_str = row["emotion"].strip("[]")
            emotion = [int(x) for x in emotion_str.split()]

            text = " ".join(dialog)
            label = emotion[0]   # allowed & simple

            self.texts.append(word_tokenize(text.lower()))
            self.labels.append(label)

        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab

    def build_vocab(self, texts):
        vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2
        for tokens in texts:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab

    def encode(self, tokens):
        ids = [self.vocab.get(t, 1) for t in tokens]
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.encode(self.texts[idx]))
        y = torch.tensor(self.labels[idx])
        return x, y
