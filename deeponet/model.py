import torch
from torch import nn
import math
from efficient_kan import KAN


def create_net(sizes):
    net = []
    for i in range(len(sizes)-1):
        net.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            net.append(nn.GELU())
    return nn.Sequential(*net)


class DeepONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.branch_net = create_net(
            [hparams["num_input"]]
            + [hparams["hidden_size"]]*(hparams["branch_hidden_depth"]-1)
            + [hparams["num_branch"]]
        )
        self.trunk_net = create_net(
            [hparams["dim_output"]]
            + [hparams["hidden_size"]]*(hparams["trunk_hidden_depth"]-1)
            + [hparams["num_branch"]]
        )
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, u, y):
        window = y.shape[1]
        branch_out = self.branch_net(u)
        trunk_out = torch.stack([self.trunk_net(y[:, i:i+1])
                                for i in range(window)], dim=2)
        pred = torch.einsum("bp,bpl->bl", branch_out, trunk_out) + self.bias
        return pred


class KANON(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        num_input = hparams["num_input"]
        hidden_size = hparams["hidden_size"]
        hidden_depth = hparams["hidden_depth"]
        num_branch = hparams["num_branch"]

        self.branch_net = KAN([num_input] + [hidden_size]
                              * (hidden_depth-1) + [num_branch])
        self.trunk_net = KAN([1] + [hidden_size] *
                             (hidden_depth-1) + [num_branch])

    def forward(self, u, y):
        window = y.shape[1]
        branch_out = self.branch_net(u)
        trunk_out = torch.stack([self.trunk_net(y[:, i:i+1])
                                for i in range(window)], dim=2)
        pred = torch.einsum("bp,bpl->bl", branch_out, trunk_out)
        return pred


class Encoder(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x):
        """
        - x: (B, W, 1)
        - h_n: (D * L, B, H) (D = 2 for bidirectional)
        - c_n: (D * L, B, H) (D = 2 for bidirectional)
        """
        _, (h_n, c_n) = self.rnn(x)
        return h_n, c_n


class Decoder(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x, h_c):
        """
        - x: (B, W, 1)
        - h_c: (D * L, B, H) (D = 2 for bidirectional)
        - o: (B, W, D * H) (D = 2 for bidirectional)
        - out: (B, W, 1)
        """
        o, _ = self.rnn(x, h_c)
        out = self.fc(o)
        return out


class VAONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        hidden_size = hparams["hidden_size"]
        latent_size = hparams["latent_size"]
        num_layers = hparams["num_layers"]
        dropout = hparams["dropout"]
        kl_weight = hparams["kl_weight"]

        self.branch_net = Encoder(hidden_size, num_layers, dropout)
        self.trunk_net = Decoder(hidden_size, num_layers, dropout)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        self.fc_z = nn.Linear(latent_size, hidden_size)
        self.kl_weight = kl_weight
        self.reparametrize = True

    def forward(self, u, y):
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        (h0, c0) = self.branch_net(u)

        # Reparameterize (VAE)
        hp = h0.permute(1, 0, 2).contiguous()     # B, D * L, H
        mu = self.fc_mu(hp)                     # B, D * L, Z
        logvar = self.fc_var(hp)                # B, D * L, Z
        if self.reparametrize:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Decoding
        hz = self.fc_z(z)                       # B, D * L, H
        hzp = hz.permute(1, 0, 2).contiguous()    # D * L, B, H
        h_c = (hzp, c0)
        o = self.trunk_net(y, h_c)              # B, W2, 1
        return o.squeeze(-1), mu, logvar


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        - x: (B, W, d_model)
        - self.pe: (1, M, d_model)
        - self.pe[:, :x.size(1), :]: (1, W, d_model)
        - output: (B, W, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TFEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)

    def forward(self, x):
        """
        - x: (B, W1, 1)
        - x (after embedding): (B, W1, d_model)
        - out: (B, W1, d_model)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        return out


class TFDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, memory):
        """
        - x: (B, W2, 1)
        - x (after embedding): (B, W2, d_model)
        - memory: (B, W1, d_model)
        - out: (B, W2, d_model)
        - out (after fc): (B, W2, 1)
        - out (after squeeze): (B, W2)
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        out = self.transformer_decoder(x, memory)
        out = self.fc(out)
        out = out.squeeze(-1)
        return out


class TFONet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        d_model = hparams["d_model"]
        nhead = hparams["nhead"]
        num_layers = hparams["num_layers"]
        dim_feedforward = hparams["dim_feedforward"]
        dropout = hparams["dropout"]

        self.branch_net = TFEncoder(
            d_model, nhead, num_layers, dim_feedforward, dropout)
        self.trunk_net = TFDecoder(
            d_model, nhead, num_layers, dim_feedforward, dropout)

    def forward(self, u, y):
        """
        - u: (B, W1)
        - y: (B, W2)
        - u (after reshape): (B, W1, 1)
        - y (after reshape): (B, W2, 1)
        - memory: (B, W1, d_model)
        - o: (B, W2)
        """
        B, W1 = u.shape
        _, W2 = y.shape
        u = u.view(B, W1, 1)
        y = y.view(B, W2, 1)

        # Encoding
        memory = self.branch_net(u)

        # Decoding
        o = self.trunk_net(y, memory)
        return o
