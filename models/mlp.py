
import torch
import torch.nn as nn
from attentive_regularization.modules.attention import AttentionModule
from attentive_regularization.utils.regularization import attentive_l2_regularization

class AttentiveMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lambda_reg):
        super(AttentiveMLP, self).__init__()
        self.lambda_reg = lambda_reg

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Create attention modules for each layer
        self.attention_fc1 = AttentionModule(self.fc1.weight.numel())
        self.attention_fc2 = AttentionModule(self.fc2.weight.numel())

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def regularization_loss(self):
        # Get flattened weights
        weights_fc1 = self.fc1.weight.view(-1)
        weights_fc2 = self.fc2.weight.view(-1)

        # Get attention scores
        attention_scores_fc1 = self.attention_fc1(weights_fc1)
        attention_scores_fc2 = self.attention_fc2(weights_fc2)

        # Calculate regularization loss
        reg_loss_fc1 = attentive_l2_regularization(weights_fc1, attention_scores_fc1, self.lambda_reg)
        reg_loss_fc2 = attentive_l2_regularization(weights_fc2, attention_scores_fc2, self.lambda_reg)

        return reg_loss_fc1 + reg_loss_fc2

