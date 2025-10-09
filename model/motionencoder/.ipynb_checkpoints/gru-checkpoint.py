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
    
class GRUEncoder(pl.LightningModule):

    def __init__(self, nfeats, num_layers, latent_dim, vae, **kwargs):
        super().__init__()

        # save hyper params to self.hparams
        self.save_hyperparameters(logger=False)

        self.skelton_embedding = nn.Linear(nfeats, latent_dim)
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers=num_layers)

        if vae:
            self.mu = nn.Linear(latent_dim, latent_dim)
            self.logvar = nn.Linear(latent_dim, latent_dim)
        else:
            self.output = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, length):

        bs, n_frame, n_dim = x.shape

        x = self.skelton_embedding(x)

        x = x.permute(1, 0, 2) # [n_frame, bs, latent_dim]

        x = self.gru(x)[0]

        x = x.permute(1, 0, 2) # [bs, n_frame, latent_dim]

        x = x[tuple(
            torch.stack((torch.arange(bs, device=x.device), torch.tensor(length, device=x.device)-1))
        )]

        if self.hparams.vae:
            mu = self.mu(x)
            logvar = self.logvar(x)
            std = logvar.exp().pow(0.5)
            return torch.distributions.Normal(mu, std)
        
        else:
            return self.output(x)




if __name__ == "__main__":


    pe = PositionalEncoding(126, 5000, False)

    # test GRUEncoder
    n_feats = 14
    n_layer = 1
    latent_dim = 256
    encoder = GRUEncoder(n_feats, n_layer, latent_dim, True)
    x = torch.randn(3, 10, n_feats)
    lengths = [5, 7, 9]
    output = encoder(x, lengths)
    print(output.shape)