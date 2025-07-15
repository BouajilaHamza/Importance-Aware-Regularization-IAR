import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, num_weights, hidden_dim=128):
        super(AttentionModule, self).__init__()
        self.num_weights = num_weights
        # Use a fixed hidden dimension instead of num_weights // 2
        self.fc1 = nn.Linear(num_weights, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_weights)
        self.sigmoid = nn.Sigmoid()

    def forward(self, weights):
        # weights is a flattened tensor of the weights of a layer
        attention_scores = self.sigmoid(self.fc2(self.relu(self.fc1(weights))))
        return attention_scores

