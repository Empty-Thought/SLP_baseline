import torch
import torch.nn as nn
import pytorch_lightning as pl

def lengths_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  # [B, T]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, batch_first=False, dropout=0.1):
        super().__init__()

        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # div_term [d_model // 2]
 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):

        if self.batch_first:
            # x [batch, len, d_model]

            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            # x [len, batch, d_model]
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class GRUDecoder(pl.LightningModule):
    def __init__(self, n_feats, latent_dim, num_layers, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.output_dim = n_feats
        self.embedding_layer = nn.Linear(latent_dim + 1, latent_dim)

        self.gru = nn.GRU(latent_dim, latent_dim, num_layers=num_layers) # if batch_first = False, input [T, B, D]
        self.proj = nn.Linear(latent_dim, n_feats)

    def forward(self, z, lengths):

        mask = lengths_to_mask(lengths, z.device)  # [B, T]
        bs, n_frame = mask.shape

        n_feats = self.hparams.n_feats

        lengths = torch.tensor(lengths, device=z.device)

        z = z[None].repeat((n_frame, 1, 1)) 

        # time information
        time = mask * 1 / (lengths[..., None] - 1)

        time = (time[:, None] * torch.arange(time.shape[1], device=z.device))[:, 0]

        time = time.T[..., None]
        z = torch.cat((z, time), 2)

        z = self.embedding_layer(z)

        z = self.gru(z)[0]  # [T, B, D]
        output = self.proj(z)  # [T, B, n_feats]
        # print(~mask.T)

        output[~mask.T] = 0
        output = output.permute(1, 0, 2)  # [B, T, n_feats]
        return output

if __name__ == "__main__":


    # test lengths_to_mask
    lengths = [5, 7, 9]
    mask = lengths_to_mask(lengths, device='cpu')

    # test GRUDecoder
    n_feats = 10
    latent_dim = 16
    num_layers = 1
    model = GRUDecoder(n_feats, latent_dim, num_layers)
    z = torch.randn((3, latent_dim))
    output = model(z, lengths)
    print(output.shape)  # [3, max_len, n_feats]