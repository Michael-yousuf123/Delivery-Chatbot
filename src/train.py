import json 
from src.utils import tokenization, normalize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

InputPath = '/home/miki/Desktop/gitDL/delivery-chatbot/data/inputs.json'

with open(InputPath, "r") as f:
    data = json.load(f)
print(data)
words = []
classes = []
document = []
bow = []
class ExtractEntities():
    def getExtraction(self, data):
        for input in data["inputs"]:
            for pattern in input['patterns']:
                w = tokenization(data['inputs'])
                word = [stem(normalize(w))]
                words.extend(word)
                document.append(word, input("tag"))
            if input['tag'] not in classes:
                classes.append(input['tag'])
        return words, classes, document
    
    def traindataset(self):
        y_train = []
        for (patterns, tag) in document:
            bag = bag_of_words()
            X_train = bag
            label = classes.index(tag)
            y_train.append(label)
            y_train = np.array(y_train)
        return X_train, y_train

class BotData(Dataset, ExtractEntities):
    def __init__(self):
        super(BotData, self).__init__()
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples

batch_size = 8
dataset = BotData()
train_dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)