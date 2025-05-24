import torch.nn as nn

# Probabilistic User Preference Generation Module (PUPG)

class UserNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.topic_num = input_dim
        self.output_dim = output_dim

        self.residual_block = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim)
        )

        self.output_block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, topic_vec):
        residual = self.residual_block(topic_vec)
        topic_vec = topic_vec + residual 
        return self.output_block(topic_vec)

