import numpy as np 
import torch
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from model.metrics.TEMOS_metrics import ComputeMetrices

def remove_padding(tensor, length):

    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensor, length)]



class BaseModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

    def __pose_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())
        
        self.hparams.n_trainable_params = trainable
        self.hparams.n_nontrainable_params = nontrainable
    
    def training_step(self,  batch, batch_idx):
        return self.forward_step("train", batch, batch_idx)
    

    def validation_step(self, batch, batch_idx):
        return self.forward_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.forward_step("test", batch, batch_idx)

    def allsplit_epoch_end(self, split: str):
        losses = self.losses[split]
        loss_dict = losses.compute(split)
        dico = {losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items()}

        if split == "val":
            metrics_dict = self.metrics.compute()
            dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items()})
        dico.update({"epoch": float(self.trainer.current_epoch),
                        "step": float(self.trainer.current_epoch)})
        self.log_dict(dico)

    def on_train_epoch_end(self):
        self.allsplit_epoch_end("train")
    
    def on_validation_epoch_end(self):
        self.allsplit_epoch_end("val")
    
    def on_test_epoch_end(self):
        self.allsplit_epoch_end("test")

    def configure_optimizers(self):
        return super().configure_optimizers()


class TEMOS(BaseModel):

    def __init__(self, textencoder, motionencoder, motiondecoder, losses, optim, transforms, nfeats, vae, latent_dim, **kwargs):

        super().__init__()

        self.textencoder = instantiate(textencoder)
        self.motionencoder = instantiate(motionencoder, nfeats=nfeats)
        self.motiondecoder = instantiate(motiondecoder, n_feats=nfeats)
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct
        # TODO: optim
        self.optimizer = instantiate(optim, params=self.parameters())

        self._loss = torch.nn.ModuleDict({split: instantiate(losses, vae=vae, _recursive_=False) for split in ['loss_train', 'loss_val', 'loss_test']})
        self.losses = {key: self._loss["loss_" + key] for key in ['train', 'val', 'test']}

        # TODO: metrics
        self.metrics = ComputeMetrices()

    def forward(self, batch):
        data_from_text = self.text_to_motion_forward(batch['text'], batch['lengths'])

        # need remove padding

        return remove_padding(data_from_text.joints, batch['lengths'])
    
    def sample_from_distribution(self, distribution, *, factor=None, sample_mean=False):

        if sample_mean:
            return distribution.loc
        
        if factor is None:
            return distribution.rsample()

        eps = distribution.rsample() - distribution.loc
        return distribution.loc + eps * factor


    def text_to_motion_forward(self, text_sentences, lengths, return_latent=False):

        if self.hparams.vae:
            distribution = self.textencoder(text_sentences)
            latent_vector = self.sample_from_distribution(distribution)
        else:
            distribution = None
            latent_vector = self.textencoder(text_sentences)
        
        motion_output = self.motiondecoder(latent_vector, lengths)
        datastruct = self.Datastruct(features=motion_output)
        if not return_latent:
            return datastruct
        return datastruct, latent_vector, distribution

    def motion_to_motion_forward(self, data_input, lengths, return_latent=False):

        if self.hparams.vae:
            distribution = self.motionencoder(data_input.features, lengths)
            latent_vector = self.sample_from_distribution(distribution)
        
        else:
            distribution = None
            latent_vector = self.motionencoder(data_input.features, lengths)
        
        features = self.motiondecoder(latent_vector, lengths)
        # print(features.device)
        datastruct = self.Datastruct(features=features)

        if not return_latent:
            return datastruct   
        return datastruct, latent_vector, distribution

    def forward_step(self, split, batch, batch_idx):

        ret1 = self.text_to_motion_forward(batch['text'], batch['length'], return_latent=True)
        datastruct_from_text, latent_from_text, distribution_text = ret1

        ret2 = self.motion_to_motion_forward(batch['datastruct'], batch['length'], return_latent=True)
        datastruct_from_motion, latent_from_motion, distribution_motion = ret2

        # GT data
        datastruct_ref = batch['datastruct']
        # print(datastruct_ref.rfeats.device)
        mu_ref = torch.zeros_like(distribution_text.loc)
        scale_ref = torch.ones_like(distribution_text.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        # TODO
        loss = self.losses[split].update(
            ds_text=datastruct_from_text, 
            ds_motion=datastruct_from_motion, 
            ds_ref=datastruct_ref, 
            lat_text=latent_from_text, 
            lat_motion=latent_from_motion, 
            dis_text=distribution_text, 
            dis_motion=distribution_motion, 
            dis_ref=distribution_ref
        )



        if split == "val":
            # Compute the metrics
            self.metrics.update(datastruct_from_text.detach().features,
                                datastruct_ref.detach().features,
                                batch["length"])


        return loss
    