import torch
import torch.nn as nn
import numpy as np
from config import ACTIVATION_FUNCTIONS

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class FeatureEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size=3, max_len=5000):
        super(FeatureEmbedding, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.positional_encoding = PositionalEncoding(d_model, max_len)

    def forward(self, x):
        # x shape: (batch_size, input_dim, seq_len)
        x = self.cnn(x)  # shape: (batch_size, d_model, seq_len)
        x = self.relu(x)
        x = self.positional_encoding(x.permute(2, 0, 1))  # shape: (seq_len, batch_size, d_model)
        return x
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q = self.query(q).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = self.softmax(scores)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(context.size(0), -1, self.d_model)
        return self.out(context)

class InterModalityAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(InterModalityAttention, self).__init__()
        self.attention = ScaledDotProductAttention(d_model, num_heads)

    def forward(self, modalities):
        # modalities: List of tensors with shape (seq_len, batch_size, d_model)
        combined = torch.cat(modalities, dim=0)  # (num_modalities * seq_len, batch_size, d_model)
        return self.attention(combined, combined, combined)

class IntraModalityAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(IntraModalityAttention, self).__init__()
        self.attention = ScaledDotProductAttention(d_model, num_heads)

    def forward(self, modality):
        # modality: Tensor with shape (seq_len, batch_size, d_model)
        return self.attention(modality, modality, modality)

class ResidualEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, multimodal=True):
        super(ResidualEncoderBlock, self).__init__()
        self.multimodal = multimodal
        if self.multimodal:
            self.inter_ma = InterModalityAttention(d_model, num_heads)
        self.intra_ma = IntraModalityAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.residual = nn.Identity()

    def forward(self, modalities):
        if self.multimodal:
            inter_ma_output = self.inter_ma(modalities)
            inter_ma_output = self.norm1(inter_ma_output)
        else:
            inter_ma_output = modalities[0]  # For unimodal, just use the single modality directly
        
        intra_ma_outputs = [self.intra_ma(modality) for modality in modalities]
        intra_ma_outputs = [self.norm2(output) + modality for output, modality in zip(intra_ma_outputs, modalities)]
        
        ffn_outputs = [self.ffn(output) + output for output in intra_ma_outputs]
        return ffn_outputs

class OutModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.5):
        super(OutModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        attn_output, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = attn_output.squeeze(1)
        x = self.fc2(x)
        return x

class UnimodalModel(nn.Module):
    def __init__(self, params, kernel_size=3, max_len=5000):
        super(UnimodalModel, self).__init__()
        self.feature_embedding = FeatureEmbedding(params.d_in[0], params.model_dim, kernel_size, max_len)
        self.encoder = nn.ModuleList([ResidualEncoderBlock(params.model_dim, params.nhead, multimodal=False) for _ in range(params.encoder_n_layers)])
        self.out = OutModule(params.model_dim, params.d_fc_out, params.n_targets, params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = x[0]
        # x shape: (batch_size, input_dim, seq_len)
        embedded_feature = self.feature_embedding(x)  # shape: (seq_len, batch_size, d_model)
        
        encoded_feature = [embedded_feature]
        for layer in self.encoder:
            encoded_feature = layer(encoded_feature)

        encoded_feature = encoded_feature[0]
        # Ensure the encoded feature has the correct dimensions
        batch_size = x.size(0)
        encoded_feature = encoded_feature.permute(1, 0, 2).contiguous().view(batch_size, -1, encoded_feature.size(-1))
        
        # Extract the last time step's features
        last_timestep_feature = encoded_feature[:, -1, :]  # shape: (batch_size, d_model)
        
        y = self.out(last_timestep_feature)
        activation = self.final_activation(y)
        return activation, last_timestep_feature
    
class MultimodalModel(nn.Module):
    def __init__(self, params, kernel_size=3, max_len=5000):
        super(MultimodalModel, self).__init__()
        self.params = params
        
        d_encoder_out = params.model_dim * len(params.d_in)
        
        self.feature_embeddings = nn.ModuleList([
            FeatureEmbedding(input_dim, params.model_dim, kernel_size, max_len) for input_dim in params.d_in
        ])
        self.encoder = nn.ModuleList([ResidualEncoderBlock(params.model_dim, params.nhead) for _ in range(params.encoder_n_layers)])
        self.out = OutModule(d_encoder_out, params.d_fc_out, params.n_targets, dropout_rate=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, inputs, x_len):
        # inputs: List of tensors with shape (batch_size, input_dim, seq_len)
        embedded_features = [embedding(input) for embedding, input in zip(self.feature_embeddings, inputs)]
        
        encoded_features = embedded_features
        for layer in self.encoder:
            encoded_features = layer(encoded_features)

        # Ensure the encoded features have the correct dimensions
        batch_size = inputs[0].size(0)
        encoded_features = [feature.permute(1, 0, 2).contiguous().view(batch_size, -1, feature.size(-1)) for feature in encoded_features]

        # Extract the last time step of each modality's features and concatenate them
        concatenated_features = torch.cat([feature[:, -1, :] for feature in encoded_features], dim=-1)  

        y = self.out(concatenated_features)
        activation = self.final_activation(y)
        return activation, concatenated_features

