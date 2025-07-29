import torch
import torch.nn as nn


class MiniTransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # expand in this case 16 * 4 = 64
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim) # compress back to 16
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x) # self attention Q K V. Since this is self attention it's all x, it will be different on cross attention
        x = self.norm1(x + attn_output) # Residual then normalize
        ffn_output = self.ffn(x) # ffn network
        x = self.norm2(x + ffn_output) # Residual then normalize
        return x


# 1 sentence (batch), 5 words (tokens), each word is a 16-number vector.
x = torch.randn(1, 5, 16)  # batch_size=1, seq_len=5, embed_dim=16
model = MiniTransformerBlock(embed_dim=16)
out = model(x) # py torch nn.Module inheritance __call__ a forward function when used like this
print(out.shape)