import torch 
import torch.nn as nn

class ConvolutionNN(nn.Module):

    def __init__(self, input_size, hidden_layer, class_number):
        super(ConvolutionNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, hidden_layer)
        self.l3 = nn.Linear(hidden_layer, class_number)
        self.relu = nn.ReLu()

    def forward(self, x):
        out = self.l1(x)
        out =self.relu(out)
        out = self.l2(out)
        out =self.relu(out)
        out = self.l3(out)
        return out


    