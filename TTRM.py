import torch.nn as nn
from torch.nn import init

# (c) Transformer-based Trajectory Representation Module (TTRM)

class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

class TransEncoder(BaseEncoder):
    def __init__(self, config):
        super().__init__()
        input_dim = config.Embedding.base_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4,
            dim_feedforward=input_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        encoder_norm = nn.LayerNorm(input_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
            norm=encoder_norm
        )

        self.initialize_parameters()

    def forward(self, embedded_out, src_mask):
        return self.encoder(embedded_out, mask=src_mask)
