import torch
import torch.nn as nn
from .sequence_loader import EmbeddingSequencePreparer

class PositionalEncoding(nn.Module):
    """
    Standard additive sinusoidal positional encodings for the Transformer baseline.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, L, D)
        return x + self.pe[:, :x.size(1)]


class LSTMClassifier(nn.Module):
    """
    Standard LSTM Classifier Baseline.
    Processes the raw sequence after projecting it to d_model feature space.
    """
    def __init__(self, in_channels=17, embedding_dim=128, num_layers=2, num_classes=4, dropout=0.1):
        super().__init__()
        self.preparer = EmbeddingSequencePreparer(in_channels, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, L, C_in)
        proj = self.preparer(x)  # (B, L, D)
        
        # LSTM output shape: (B, L, D)
        lstm_out, _ = self.lstm(proj)
        
        # Mean pool over sequence dimension
        season_embedding = torch.mean(lstm_out, dim=1)  # (B, D)
        
        logits = self.classifier_head(season_embedding)
        return logits


class TransformerClassifier(nn.Module):
    """
    Standard Attention-based Transformer Encoder Classifier Baseline.
    Projects raw features, injects positional encodings, and runs self-attention layers.
    """
    def __init__(self, in_channels=17, embedding_dim=128, num_layers=2, nhead=8, num_classes=4, dropout=0.1):
        super().__init__()
        self.preparer = EmbeddingSequencePreparer(in_channels, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, L, C_in)
        proj = self.preparer(x)  # (B, L, D)
        pos = self.pos_encoder(proj)  # (B, L, D)
        
        # Transformer encoder output shape: (B, L, D)
        trans_out = self.transformer_encoder(pos)
        
        # Mean pool over sequence dimension
        season_embedding = torch.mean(trans_out, dim=1)  # (B, D)
        
        logits = self.classifier_head(season_embedding)
        return logits

if __name__ == "__main__":
    # Test baselines shape
    lstm = LSTMClassifier(17, 64, 2, 4)
    trans = TransformerClassifier(17, 64, 2, 8, 4)
    
    x = torch.randn(8, 6, 17)
    print("LSTM logits shape:", lstm(x).shape)          # Should be (8, 4)
    print("Transformer logits shape:", trans(x).shape)  # Should be (8, 4)
