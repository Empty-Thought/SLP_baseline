import torch
from torch.nn import Module
import hydra
class KLLoss:
    def __init__(self):
        pass

    def __call__(self, p, q):
        div = torch.distributions.kl_divergence(p, q)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"
    
class KLLossMulti:
    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, plist, qlist):
        return sum(self.klloss(p, q) for p, q in zip(plist, qlist))
    
    def __repr__(self):
        return "KLLossMulti()"
    
# torch.nn.MSELoss(reduction='mean')
# torch.nn.SmoothL1Loss(reduction='mean')

class TemosComputeLoss(Module):

    def __init__(self, vae, mode=None, **kwargs):
        super().__init__()

        self.vae = vae
        losses = []
        
        
        # rfeats => rfeats
        losses.append("recons_rfeats2rfeats")
        # text => rfeats
        losses.append("recons_text2rfeats")

        # KL Losses
        kl_losses = []
        if vae:
            kl_losses.extend(["kl_text2motion", "kl_motion2text"])
            kl_losses.extend(["kl_text"])
            kl_losses.extend(["kl_motion"])
            losses.extend(kl_losses)
        if not vae:
            losses.append("latent_manifold")

        # total loss
        losses.append("total")
        
        for loss in losses:
            self.register_buffer(loss, torch.tensor(0.0))

        self.register_buffer("count", torch.tensor(0.0))

        self.losses = losses    

        self._losses_func = {
            loss: hydra.utils.instantiate(kwargs[loss + '_func']) for loss in losses if loss != "total"
        }
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}

    def _update_loss(self, loss_name, outputs, inputs):

        '''
        val = weight * loss_func(outputs, inputs)
        return: val
        '''
        val = self._losses_func[loss_name](outputs, inputs)
        getattr(self, loss_name).__iadd__(val.detach())
        weighted_loss = self._params[loss_name] * val
        return weighted_loss


    def update(self, ds_text, ds_motion, ds_ref, lat_text, lat_motion, dis_text, dis_motion, dis_ref):

        total = 0.0

        # # jfeats: xyz features
        # recons_jfeats2jfeats_loss = self._update_loss("recons_jfeats2jfeats", ds_motion.jfeats, ds_ref.jfeats)
        # recons_text2jfeats_loss = self._update_loss("recons_text2jfeats", ds_text.jfeats, ds_ref.jfeats)

        # rfeats: rotation features
        recons_rfeats2rfeats_loss = self._update_loss("recons_rfeats2rfeats", ds_motion.features, ds_ref.features)
        total += recons_rfeats2rfeats_loss
        recons_text2rfeats_loss = self._update_loss("recons_text2rfeats", ds_text.features, ds_ref.features)
        total += recons_text2rfeats_loss

        if self.vae:
            kl_text2motion_loss = self._update_loss("kl_text2motion", dis_text, dis_motion)
            total += kl_text2motion_loss
            kl_motion2text_loss = self._update_loss("kl_motion2text", dis_motion, dis_text)
            total += kl_motion2text_loss
            kl_text = self._update_loss("kl_text", dis_text, dis_ref)
            total += kl_text
            kl_motion = self._update_loss("kl_motion", dis_motion, dis_ref)
            total += kl_motion
        # latent space loss if not vae
        if not self.vae:
            latent_manifold_loss = self._update_loss("latent_manifold", lat_text, lat_motion)
            total += latent_manifold_loss

        self.total += total.detach()
        self.count += 1.0
        return total

    def compute(self, split):
        count = self.count
        return {
            loss : getattr(self, loss) / count for loss in self.losses
        }  
    
    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name