import logging

import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

def load_checkpoint(model, checkpoint_path, *, eval_mode):


    import torch
    # model is pl.LightningModule
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    if eval_mode:
        model.eval()
        logger.info('Model set to eval mode')


@hydra.main(version_base=None, config_path='config', config_name='sample')
def _sample(cfg: DictConfig):
    return sample(cfg)


def sample(newcfg):

    # cfg may get from newcfg and oldcfg
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    ckpt_path = newcfg.ckpt_path

    previous_cfg = OmegaConf.load(output_dir / '.hydra/config.yaml')

    cfg = OmegaConf.merge(previous_cfg, newcfg)

    storage_path = output_dir / "samples"

    storage_path = Path(storage_path)

    storage_path.mkdir(parents=True, exist_ok=True)

    import pytorch_lightning as pl
    import numpy as np
    import torch
    from hydra import initialize

    data_module = initialize(cfg.data)
    logger.info(f"Data module {cfg.data.dataname} loaded")

    model = initialize(
        cfg.model, nfeats=data_module.nfeats, nvids_to_save=None, _recursive_=False
    )

    # load last checkpoint
    load_checkpoint(model, ckpt_path, eval_mode=True)

    dataset = getattr(data_module, f"{cfg.split}_dataset")

    from datastruct import collate_data_and_text

    from rich.progress import Progress, track
    with torch.no_grad():
        with Progress(transient=True) as progress:
            task = progress.add_task("Sampling", total=len(dataset.keyids))

            for keyid in dataset.keyids:
                progress.update(task, description=f"Sampling {keyid}...")
                
                # it supports multiple sample and generate for each input
                for index in range(cfg.number_of_samples):

                    one_data = dataset.load_keyid(keyid)
                    batch = collate_data_and_text([one_data])
                    pl.seed_everything(cfg.seed + index)

                    joints = model(batch)[0]
                    joints = joints.cpu().numpy()
                    # resample if necessary

                    if cfg.number_of_samples > 1:
                        npy_save_path = storage_path / f"{keyid}_{index}.npy"
                    else:
                        npy_save_path = storage_path / f"{keyid}.npy"
                    
                    np.save(npy_save_path, joints)
                progress.update(task, advance=1)
    logger.info(f"All samples are saved to {storage_path}")

if __name__ == "__main__":
    _sample()