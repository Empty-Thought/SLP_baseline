import os
import sys
import numpy as np

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

def get_split_keyids(path, split):
    from pathlib import Path
    filepath = Path(path) / (split + '.txt')
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")

def extend_paths(path, keyids, *, onesample=True, number_of_samples=1):

    if number_of_samples ==1:
        template_path = str(path / "KEYID_INDEX.npy")
        paths = [template_path.replace("INDEX", str(index)) for index in range(number_of_samples)]
    else:
        paths = [str(path / "KEYID.npy")]
    
    all_paths = []
    for path in paths:
        all_paths.extend([path.replace("KEYID", keyid) for keyid in keyids])
    return all_paths


@hydra.main(version_base=None, config_path="configs", config_name="skeleton_render")
def _render(cfg: DictConfig):
    return render(cfg)


def plot_joints(SMPLX_joints):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = SMPLX_joints[:, 0]
    ys = SMPLX_joints[:, 1]
    zs = SMPLX_joints[:, 2]
    ax.scatter(xs, ys, zs)
    

    pass

def regress_joints(SMPLX_poses, *, SMPLX_NEUTRAL_PATH):

    from smplx.body_models import SMPLX
    SMPLX_Layer = SMPLX(SMPLX_NEUTRAL_PATH, ext='npz', use_pca=False).eval()

    body_pose = SMPLX_poses[None, 0:22, :]
    right_hand_pose = SMPLX_poses[None, 22:37, :]
    left_hand_pose = SMPLX_poses[None, 37:52, :]

    out = SMPLX_Layer(
        body_pose=body_pose,
        right_hand_pose=right_hand_pose,
        left_hand_pose=left_hand_pose
    )
    return out.joints[0].cpu().numpy()


def render(cfg: DictConfig) -> None:
    if cfg.npy is None:
        from pathlib import Path
        keyids = get_split_keyids(path=Path(cfg.path.dataset), split=cfg.split)

        path = Path(cfg.folder)

        paths = extend_paths(path, keyids)

    else:
        paths = [cfg.npy]


    for path in paths:

        try:
            data = np.load(path)
        except FileNotFoundError:
            logger.warning(f"File {path} not found, skipping")
            continue

        frames_folder = path.replace('.npy', '.png')

