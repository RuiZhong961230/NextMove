import torch.nn as nn

# Next Location Feature Aggregation Module (NLFA)

class MyFullyConnect(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
        )

        self.post_process = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.residual_block(x) + x  
        out = self.post_process(out)
        return self.classifier(out)
