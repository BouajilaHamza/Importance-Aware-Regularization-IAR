import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, num_weights):
        super(AttentionModule, self).__init__()
        self.num_weights = num_weights
        self.fc1 = nn.Linear(num_weights, num_weights // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_weights // 2, num_weights)
        self.sigmoid = nn.Sigmoid()

    def forward(self, weights):
        # weights is a flattened tensor of the weights of a layer
        attention_scores = self.sigmoid(self.fc2(self.relu(self.fc1(weights))))
        return attention_scores

