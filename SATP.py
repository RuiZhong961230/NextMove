import torch
from torch import nn

# (b) Self-attention-based Arrival Time Prediction Module (STAP)

class ArrivalTime(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_dim = config.Embedding.base_dim
        self.num_heads = 4
        self.head_dim = self.base_dim // self.num_heads
        self.num_users = config.Dataset.num_users
        self.timeslot_num = 24

        if config.Model.at_type == 'attn':
            self.user_preference = nn.Embedding(self.num_users, self.base_dim)
            self.w_q = nn.ModuleList([nn.Linear(self.base_dim * 2, self.head_dim) for _ in range(self.num_heads)])
            self.w_k = nn.ModuleList([nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.w_v = nn.ModuleList([nn.Linear(self.base_dim, self.head_dim) for _ in range(self.num_heads)])
            self.unify_heads = nn.Linear(self.base_dim, self.base_dim)

    def _compute_attention(self, user_feature, time_feature, timeslot_embedded, hour_mask):
        query = torch.cat([user_feature, time_feature], dim=-1)
        key = timeslot_embedded
        head_outputs = []

        for i in range(self.num_heads):
            q = self.w_q[i](query)
            k = self.w_k[i](key)
            v = self.w_v[i](key)

            attn_scores = torch.matmul(q, k.T) / (k.size(-1) ** 0.5)
            attn_scores = attn_scores.masked_fill(hour_mask == 1, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)

            head_outputs.append(torch.matmul(attn_weights, v))

        return torch.cat(head_outputs, dim=-1)

    def forward(self, timeslot_embedded, batch_data):
        user_x = batch_data['user']
        hour_x = batch_data['hour']
        batch_size, seq_len = hour_x.shape
        hour_flat = hour_x.view(batch_size * seq_len)
        hour_mask = batch_data['hour_mask'].view(batch_size * seq_len, -1)

        if self.config.Model.at_type == 'truth':
            return timeslot_embedded[batch_data['timeslot_y']]

        if self.config.Model.at_type == 'attn':
            user_feature = self.user_preference(user_x).unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, self.base_dim)
            time_feature = timeslot_embedded[hour_flat]
            attn_output = self._compute_attention(user_feature, time_feature, timeslot_embedded, hour_mask)
            return self.unify_heads(attn_output).view(batch_size, seq_len, -1)

        if self.config.Model.at_type == 'static':
            prob_mat = batch_data['prob_matrix_time_individual']
            weighted_time_emb = torch.matmul(prob_mat, timeslot_embedded)
            idx = torch.arange(batch_size).unsqueeze(1).expand_as(hour_x)
            return weighted_time_emb[idx, hour_x]
