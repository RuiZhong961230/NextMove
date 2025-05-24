import math
import torch
import torch.nn as nn

from PUPG import UserNet          
from SATP import ArrivalTime              
from TTRM import TransEncoder          
from NLFA import MyFullyConnect                


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))

        pos_encoding = torch.zeros(max_len, emb_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pos_encoding[:, :seq_len].detach())


class MyEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.Embedding.base_dim
        self.user_embedding = nn.Embedding(config.Dataset.num_users, dim)
        self.location_embedding = nn.Embedding(config.Dataset.num_locations, dim)
        self.timeslot_embedding = nn.Embedding(24, dim)

    def forward(self, batch_data):
        device = batch_data['location_x'].device
        user_ids = torch.arange(self.user_embedding.num_embeddings, device=device)
        timeslot_ids = torch.arange(self.timeslot_embedding.num_embeddings, device=device)

        loc_emb = self.location_embedding(batch_data['location_x'])  # [B, T, D]
        time_emb = self.timeslot_embedding(timeslot_ids)             # [24, D]
        user_emb = self.user_embedding(user_ids)                     # [U, D]

        return loc_emb, time_emb, user_emb


class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.Embedding.base_dim
        self.topic_num = config.Dataset.topic_num

        self.embedding_layer = MyEmbedding(config)
        self.encoder_type = config.Encoder.encoder_type
        self.use_at = config.Model.at_type != 'none'
        self.use_topic = self.topic_num > 0

        if self.encoder_type == 'trans':
            self.positional_encoding = PositionalEncoding(dim)
            self.encoder = TransEncoder(config)
        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

        if self.use_at:
            self.at_net = ArrivalTime(config)

        if self.use_topic:
            self.user_net = UserNet(input_dim=self.topic_num, output_dim=dim)

        # final fc input: encoder_out + at (optional) + user + topic (optional)
        fc_input_dim = dim + dim  # encoder + user
        if self.use_at:
            fc_input_dim += dim
        if self.use_topic:
            fc_input_dim += dim

        self.fc_layer = MyFullyConnect(input_dim=fc_input_dim, output_dim=config.Dataset.num_locations)
        self.out_dropout = nn.Dropout(0.1)

    def _build_future_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask.masked_fill(mask, float('-inf'))

    def forward(self, batch_data):
        user_x = batch_data['user']
        loc_x = batch_data['location_x']
        hour_x = batch_data['hour']
        batch_size, seq_len = loc_x.size()

        loc_emb, timeslot_emb, user_emb_all = self.embedding_layer(batch_data)
        time_emb = timeslot_emb[hour_x]
        seq_input = loc_emb + time_emb  # e^l + e^t

        if self.encoder_type == 'trans':
            mask = self._build_future_mask(seq_len, seq_input.device)
            encoder_out = self.encoder(self.positional_encoding(seq_input * math.sqrt(self.config.Embedding.base_dim)), src_mask=mask)
        else:
            encoder_out = self.encoder(seq_input)

        encoder_out = encoder_out + seq_input
        user_emb = user_emb_all[user_x].unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([encoder_out, user_emb], dim=-1)

        if self.use_at:
            at_emb = self.at_net(timeslot_emb, batch_data)
            combined = torch.cat([combined, at_emb], dim=-1)

        if self.use_topic:
            topic_input = batch_data['user_topic_loc']
            topic_emb = self.user_net(topic_input).unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.cat([combined, topic_emb], dim=-1)

        out = self.fc_layer(combined.view(batch_size * seq_len, combined.size(-1)))
        return out