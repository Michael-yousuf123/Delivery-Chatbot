import json
from pickletools import optimize 
from src.utils import tokenization, normalize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from predict import ConvolutionNN

InputPath = '/home/miki/Desktop/gitDL/delivery-chatbot/data/inputs.json'

with open(InputPath, "r") as f:
    data = json.load(f)
print(data)

words = []
classes = []
document = []
bow = []

for input in data["inputs"]:
    for pattern in input['patterns']:
        w = tokenization(data['inputs'])
        word = [stem(normalize(w))]
        words.extend(word)
        document.append(word, input("tag"))
    if input['tag'] not in classes:
        classes.append(input['tag'])

X_train = []
y_train =  []
for (patterns, tag) in document:
    bag = bag_of_words()
    X_train.append(bag)
    label = classes.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class BotData(Dataset):
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
input_size = len(words)
hidden_layer = 8
output_size = len(classes)
learning_rate= 0.001
num_epoches = 1000


dataset = BotData()
train_dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvolutionNN(input_size=input_size, hidden_layer=hidden_layer, output_size=output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoches in range(num_epoches):
    for (words, labels) in train_dataset:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoches + 1) % 100 == 0:
        print(f'epoch {epoches + 1}/{num_epoches}, loss = {loss.item():.4f}')
print(f'final loss, loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_layer": hidden_layer,
    "words":words,
    "classes": classes
}

FILE = "output.pth"
torch.save(data, FILE)