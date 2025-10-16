import numpy as np
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from rich.progress import track
from pathlib import Path

from datastruct import collate_data_and_text, RotTransDatastruct


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, batchsize, num_workers):
        super().__init__()

        self.dataloader_options = {
            "batch_size": batchsize,
            "num_workers": num_workers,
            "collate_fn": collate_data_and_text
        }

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.Dataset(**sample_params)
    
    def __getattr__(self, item):
        # train_dataset/val_dataset etc cached like properties
        if item.endswith("_dataset") and not item.startswith("_"):
            subset = item[:-len("_dataset")]
            item_c = "_" + item
            if item_c not in self.__dict__:
                self.__dict__[item_c] = self.Dataset(split=subset, **self.hparams)
            return getattr(self, item_c)
        classname = self.__class__.__name__
        raise AttributeError(f"'{classname}' object has no attribute '{item}'")

    def setup(self, stage):
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_options)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, **self.dataloader_options)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.dataloader_options)

class SMPLXDataModule(BaseDataModule):
    def __init__(self, data_dir, batchsize, num_workers, **kwargs):
        super().__init__(batchsize, num_workers)

        self.save_hyperparameters(logger=False)
        self.Dataset = SignDataset

        sample_overrides = {
            "split": "train", "tiny": True, "progress_bar": False
        }
        self._sample_set = self.get_sample_set(sample_overrides)

        self.nfeats = self._sample_set.nfeats
        self.transforms = self._sample_set.transforms

def get_split_keyids(path, split):
    filepath = Path(path) / (split + '.txt')
    # print(filepath)

    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")


class SignDataset(Dataset):

    def __init__(self, data_path, split_path, split, transforms=None,
                 sampler=None, framerate=None, progress_bar=False, 
                 downsample=False, tiny=False, **kwargs):

        super().__init__()

        self.split = split
        self.downsample = downsample
        self.transforms = transforms

        self.sampler = sampler
        # 
        keyids = get_split_keyids(split_path, split)

        self.split_index = list(keyids)

        enumerator = enumerate(keyids)

        if tiny:
            maxdata = 2
        else:
            maxdata = np.inf

        datapath = Path(data_path)

        motion_data = {}
        texts_data = {}
        duration_data = {}

        for i, keyid in enumerator:
            if len(motion_data) >= maxdata:
                break
            # load annotation data 
            ann_data = load_annotation(keyid, datapath)
            # load smplx data
            smplx_data = load_smplx(keyid, datapath)
            # sample smplx data
            smplx_data, duration = sample_smplx(smplx_data, framerate=framerate, downsample=self.downsample)
            # duration select, if duration less than minimum, should skip this sample

            # convert axis angle to other format if necessary
            smplx_data = smpl_data_to_matrix_and_trans(smplx_data)
            
            # conver data to other features if necessary
            # smplx_feature = self.transforms(smplx_data)
            from einops import rearrange
            smplx_feature = rearrange(smplx_data.rots, "... joints rot -> ... (joints rot)")

            motion_data[keyid] = smplx_feature
            texts_data[keyid] = ann_data
            duration_data[keyid] = duration

        self.motion_data = motion_data
        self.texts_data = texts_data
        self.keyids = list(motion_data.keys())
        self.duration_data = duration_data
        self.nfeats = len(self[0]['datastruct'].features[0])

    def _load_datastructure(self, keyid, frame_idx=None):

        features = self.motion_data[keyid]  
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct
    
    def _load_text(self, keyid):

        sequences = self.texts_data[keyid]

        # if multiple texts exist, select random one
        index = 0

        return sequences[0]

    def load_keyid(self, keyid):
        num_frames = self.duration_data[keyid]
        frame_idx = self.sampler(num_frames)

        datastruct = self._load_datastructure(keyid, frame_idx)
        text = self._load_text(keyid)

        element = {
            "datastruct": datastruct,
            "text": text,
            "keyid": keyid,
            "length": len(datastruct)
        }
        return element
    
    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self.split_index)

def load_annotation(keyid, datapath, tag="raw text from sign book"):

    text_datapath = datapath / "text"

    metapath = text_datapath / (keyid + "/meta.json")
    metadata = json.load(metapath.open())[tag]

    if metadata["num_description"] == 0:
        return None

    annpath = text_datapath / (keyid + "/text.json")
    annotations = json.load(annpath.open())[tag]

    return annotations

def load_smplx(keyid, datapath, tag="52d_pose_confidence_0.9"):

    datapath = datapath / "motion"

    smplx_datapath = datapath / (keyid + "/motion.json")

    try:
        smplx_data = json.load(smplx_datapath.open())[tag]
        smplx_data = np.array(smplx_data)
        return smplx_data
    except FileNotFoundError:
        return None



def sample_smplx(smplx_data, *, framerate, downsample):

    nframes_total = len(smplx_data)
    framerate_src = None

    if downsample:
        pass

    else:
        frames = np.arange(nframes_total)

    duration = len(frames)

    smplx_data = {
        "poses": torch.from_numpy(smplx_data[frames, 1:]).float(),
        "trans": torch.from_numpy(smplx_data[frames, 0]).float()
    }
    return smplx_data, duration



def smpl_data_to_matrix_and_trans(smplx_data):

    trans = smplx_data["trans"]
    nframes = len(trans)

    axis_angle_poses = smplx_data["poses"]
    axis_angle_poses = axis_angle_poses.reshape(nframes, -1, 3)

    # matrix poses
    return RotTransDatastruct(rots=axis_angle_poses, trans=trans)