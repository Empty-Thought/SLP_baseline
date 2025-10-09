import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from typing import List, Union
import transformers

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

class DistilbertEncoderBase(pl.LightningModule):

    def __init__(self, modelpath, finetune):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel, logging
        
        # only show err info
        logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)

        if not finetune:
            self.text_model.training = False
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.text_encode_dim = self.text_model.config.hidden_size   # 768 for distilbert-base-uncased
    
    def train(self, mode = True):

        self.training = mode

        for module in self.children():

            if module == self.text_model and not self.hparams.finetune:
                continue
            module.train(mode)
        return self

    def get_last_hidden_state(self, texts: List[str], return_mask: bool = False):
        # tokenizer can pad the sequences to the same length, and attention mask will be zero where the padding token is
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True)
        output = self.text_model(**encoded_input.to(self.text_model.device))
        if return_mask:
            return output.last_hidden_state, encoded_input.attention_mask.to(dtype=bool)
        return output.last_hidden_state

class DistilbertActorAgnosticEncoder(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        self.save_hyperparameters(logger=False)

        encoded_dim = self.text_encode_dim

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))

        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout=dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(self, texts: List[str]) -> Union[Tensor, Distribution]:
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)    # [b, seqlen, text_encode_dim], [b, seqlen]
        # each token in the sequence is encoded to a vector

        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        if self.hparams.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding

        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.hparams.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            try:
                dist = torch.distributions.Normal(mu, std)
            except ValueError as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Normal distribution creation failed: mu={mu}, std={std}, error={e}")
                # import ipdb; ipdb.set_trace()  # noqa
                # pass
            return dist
        else:
            return final[0]

if __name__ == "__main__":
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    data_path = (base_dir / "test.pkl")

    distilbert_base_uncased_path = r'D:\Workplace\Sign\\pretest\\pre_model\distilbert-base-uncased'

    model = DistilbertActorAgnosticEncoder(modelpath=distilbert_base_uncased_path,
                                           finetune=False,
                                           vae=True,
                                           latent_dim=256,
                                           ff_size=1024,
                                           num_layers=2,
                                           num_heads=2,
                                           dropout=0.1,
                                           activation="gelu")
    texts = ["Hello, my dog is cute", "Today is a good day", "I love you so much that I can't explain"]
    out = model(texts)
    # print(out)