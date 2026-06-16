import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sequence_loader import EmbeddingSequencePreparer

class MambaSSMBlock(nn.Module):
    """
    Pure PyTorch implementation of the Mamba Selective State Space Model block.
    Designed to run natively on macOS (CPU/MPS) and CUDA without binary compilation.
    Matches the exact discretization and selective scan dynamics of the Mamba paper.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Dual branches (SSM branch and Gating branch)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D Convolution over the sequence dimension (captures local temporal dependencies)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Dynamic parameter projection: maps inputs to state B and state C
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        nn.init.normal_(self.x_proj.weight, std=0.02)
        
        # Step size projection (Delta) projected directly from u_conv
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Initialize Step size projection weights
        dt_init_std = 0.02
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias to be in range [log(0.001), log(0.1)]
        dt_bias = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        self.dt_proj.bias = nn.Parameter(dt_bias)
        
        # Parameter A (S4 continuous state matrix) - Initialized in continuous space
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # Kept as log(A) to ensure stability (A < 0)
        
        # Final output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # Placeholders for tracking activations for visualization
        self.last_deltas = None
        self.last_state_activations = None

    def forward(self, x):
        # x shape: (B, L, D)
        B, L, D = x.shape
        
        # 1. Project inputs to dual branches
        projected = self.in_proj(x)  # (B, L, 2 * E)
        u, z = projected.chunk(2, dim=-1)  # u: SSM input, z: gating channel
        
        # 2. Local 1D Convolution
        u_conv = u.transpose(1, 2)  # Conv1d expects (B, E, L)
        u_conv = self.conv1d(u_conv)[:, :, :L]  # Apply padding and slice back to length L
        u_conv = u_conv.transpose(1, 2)  # Return to (B, L, E)
        u_conv = F.silu(u_conv)  # Non-linear activation (SiLU / Swish)
        
        # 3. Dynamic projections for Selection parameters B and C
        x_db = self.x_proj(u_conv)  # (B, L, N * 2)
        B_proj, C_proj = torch.split(
            x_db, [self.d_state, self.d_state], dim=-1
        )
        
        # Discretize step size (Delta) directly from u_conv (clamped to prevent recurrence explosion)
        delta = torch.clamp(F.softplus(self.dt_proj(u_conv)), min=0.0001, max=0.5)  # Shape: (B, L, E)
        
        # 4. Discretization and Recurrence (Selective Scan)
        A = -torch.clamp(torch.exp(self.A_log), min=0.1)  # Shape: (E, N) - Ensure strictly negative values with minimum decay
        
        # Latent state variable h: shape (B, E, N)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        # Save values for visual tracking in the experiments
        self.last_deltas = delta.detach().cpu()
        state_norms = []
        
        for t in range(L):
            # Extract time-step variables
            delta_t = delta[:, t, :]  # (B, E)
            B_t = B_proj[:, t, :]      # (B, N)
            C_t = C_proj[:, t, :]      # (B, N)
            u_t = u_conv[:, t, :]      # (B, E)
            
            # Dynamic discretization:
            # bar_A = exp(delta_t * A) - Shape: (B, E, N)
            bar_A = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            # bar_B = delta_t * B_t - Shape: (B, E, N)
            bar_B = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            # State Update: h_t = bar_A * h_{t-1} + bar_B * u_t
            h = bar_A * h + bar_B * u_t.unsqueeze(-1)
            
            # State Output Projection: y_t = sum_n (h_t * C_t)
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)  # (B, E)
            ys.append(y_t)
            
            # Track state norm for phenology analysis
            state_norms.append(torch.norm(h, dim=(1, 2)).detach().cpu())
            
        y = torch.stack(ys, dim=1)  # (B, L, E)
        y = F.layer_norm(y, (self.d_inner,))  # Normalize scan outputs to keep scale stable
        self.last_state_activations = torch.stack(state_norms, dim=1)  # (B, L)
        
        # 5. Gate the outputs with the gating branch (z)
        y_gated = y * F.silu(z)  # (B, L, E)
        
        # 6. Final projection back to model dimensionality
        out = self.out_proj(y_gated)  # (B, L, D)
        return out


class MambaTemporalEncoder(nn.Module):
    """
    Temporal model stack containing learned projection and multiple Mamba layers.
    Encodes variable satellite sequences into a unified "Season Embedding".
    """
    def __init__(self, in_channels=17, embedding_dim=128, num_layers=2, d_state=16):
        super().__init__()
        self.preparer = EmbeddingSequencePreparer(in_channels, embedding_dim)
        
        self.layers = nn.ModuleList([
            MambaSSMBlock(d_model=embedding_dim, d_state=d_state)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Input x: (B, L, C_in)
        # 1. Project features
        emb = self.preparer(x)  # (B, L, D_emb)
        
        # 2. Pass through Mamba layers
        out = emb
        for layer in self.layers:
            out = layer(out) + out  # Residual connection
            
        # 3. Layer normalization
        out = self.layer_norm(out)
        
        # 4. Global average pooling over the sequence dimension to create the "Season Embedding"
        season_embedding = torch.mean(out, dim=1)  # Shape: (B, D_emb)
        return season_embedding, out


class MambaClassifier(nn.Module):
    """
    Complete classifier head wrapped around the Mamba Temporal Encoder.
    """
    def __init__(self, in_channels=17, embedding_dim=128, num_layers=2, d_state=16, num_classes=4):
        super().__init__()
        self.encoder = MambaTemporalEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            d_state=d_state
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (B, L, C_in)
        season_embedding, seq_out = self.encoder(x)
        logits = self.classifier_head(season_embedding)
        return logits

if __name__ == "__main__":
    # Quick shape verification
    model = MambaClassifier(in_channels=17, embedding_dim=64, num_layers=1, num_classes=4)
    x = torch.randn(8, 6, 17)
    logits = model(x)
    print("Logits shape:", logits.shape)  # Should be (8, 4)
