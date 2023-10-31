import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        input_dim = config.num_features * config.seq_len
        hidden_dim = config.hidden_dim
        output_dim = config.output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, self.config.num_features * self.config.seq_len)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
