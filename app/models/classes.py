import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Apply self-attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # Apply feedforward
        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class Encoder(nn.Module):

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=500):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size, src_len = src.shape[0], src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, mech="add"):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.mech = mech

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        # Additive attention components
        if mech == "add":
            self.v = nn.Linear(self.head_dim, 1)
            self.W = nn.Linear(self.head_dim, self.head_dim)
            self.U = nn.Linear(self.head_dim, self.head_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for Multi-Head Attention Layer.

        Args:
            query: Query tensor of shape [batch_size, query_len, hid_dim]
            key: Key tensor of shape [batch_size, key_len, hid_dim]
            value: Value tensor of shape [batch_size, value_len, hid_dim]
            mask: Optional mask tensor to ignore certain positions

        Returns:
            x: Output tensor of shape [batch_size, query_len, hid_dim]
            attention: Attention weights of shape [batch_size, n_heads, query_len, key_len]
        """
        batch_size = query.shape[0]

        Q = self.fc_q(query)  # [batch_size, query_len, hid_dim]
        K = self.fc_k(key)    # [batch_size, key_len, hid_dim]
        V = self.fc_v(value)  # [batch_size, value_len, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.mech == "add":
            energy = self.additive_attention(Q, K)  # Additive Attention
        else:
            raise ValueError(f"Invalid attention mechanism: {self.mech}")

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention

    def additive_attention(self, Q, K):
        """
        Additive Attention Mechanism.
        Args:
            Q: Query tensor [batch_size, n_heads, query_len, head_dim]
            K: Key tensor [batch_size, n_heads, key_len, head_dim]
        Returns:
            energy: Energy tensor [batch_size, n_heads, query_len, key_len]
        """
        batch_size, n_heads, query_len, key_len, head_dim = Q.shape[0], Q.shape[1], Q.shape[2], K.shape[2], Q.shape[3]

        # Reshape Q and K to align for element-wise addition
        Q = Q.view(batch_size, n_heads, query_len, 1, head_dim)  # [batch_size, n_heads, query_len, 1, head_dim]
        K = K.view(batch_size, n_heads, 1, key_len, head_dim)    # [batch_size, n_heads, 1, key_len, head_dim]

        # Apply linear transformations and calculate energy
        energy = self.v(torch.tanh(self.W(Q) + self.U(K))).squeeze(-1)  # [batch_size, n_heads, query_len, key_len]
        return energy


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

class Decoder(nn.Module):

    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, mech, max_length=500):
        super().__init__()

        self.device = device
        self.mech = mech
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device, mech)
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        return self.fc_out(trg), attention

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, mech):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg     = self.self_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg             = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        #attention = [batch_size, n heads, trg len, src len]
        
        _trg = self.feedforward(trg)
        trg  = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        
        return trg, attention



class Seq2SeqTransformer(nn.Module):
    """
    Full Transformer Model combining Encoder and Decoder.
    """
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention


# Function for weight initialization
def initialize_weights(m):
    """
    Applies Xavier uniform initialization to all linear layers.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Embedding):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
