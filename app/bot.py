import random
import json
import torch
from src.predict import ConvolutionNN
from src.utils import bag_of_words, tokenization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
InputPath = '/home/miki/Desktop/gitDL/delivery-chatbot/data/inputs.json'
with open(InputPath, "r") as f:
    data = json.load(f)

data = torch.load()
FILE = "output.pth"


model_state = data["model_state"]
input_size = data["input_size"]
output_size = data["output_size"]
hidden_layer =  data["hidden_layer"]
words = data["words"]
classes = data["classes"]

model = ConvolutionNN(input_size=input_size, hidden_layer=hidden_layer, output_size=output_size)
model.state_dict(model_state)
model.eval()

bot_name = 'Lake Shore Resort'
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenization(sentence)
    X = bag_of_words(sentence)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy()

    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = classes[predicted.item()]

    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for input in data['inputs']:
            if tag == input['tags']:
                print(f"{bot_name}: {random.choice(input['responses'])}")
            else:
                print(f"{bot_name}: i coudn't get what you asked...")