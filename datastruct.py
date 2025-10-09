from typing import List, Dict, Optional
from torch import Tensor
from dataclasses import dataclass, fields

def collate_tensor_with_padding(batch: List[Tensor]):


    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]   # [max_T, max_D]

    size = (len(batch),) + tuple(max_size)  # (B, max_T, max_D)

    collated = batch[0].new_zeros(size)

    for i, b in enumerate(batch):
        sub_tensor = collated[i]    # [max_T, max_D]

        # narrow down each sample in collated to the size of the original sample, and add the original sample to it
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)

    return collated




@dataclass
class Datastruct:
    
    # obj[key]
    def __getitem__(self, key):
        return getattr(self, key)
    
    # obj[key] = value
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return getattr(self, key, default)
    
    # for x in obj:
    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [f.name for f in fields(self)]
        return iter(keys)
    
    def values(self):
        values = [getattr(self, f.name) for f in fields(self)]
        return iter(values)
    
    def to(self, *args, **kwargs):
        for key in self.datakeys:
            if self[key] is not None:
                self[key] = self[key].to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self[self.datakeys[0]].device

    def detach(self):
        def detach_or_none(tensor):
            if tensor is not None:
                return tensor.detach()
            return None

        kwargs = {key: detach_or_none(self[key])
                  for key in self.datakeys}
        return self.transforms.Datastruct(**kwargs)

class Transform:
    def collate(self, datastruct_list):

        example = datastruct_list[0]

        def collate_or_none(key):
            if example[key] is None:
                return None
            value_list = [x[key] for x in datastruct_list]
            return collate_tensor_with_padding(value_list)
        
        kwargs = {
            key: collate_or_none(key) for key in example.datakeys
        }

        return self.Datastruct(**kwargs)



class SMPLXTransform(Transform):

    def __init__(self, **kwargs):

        pass

    def Datastruct(self, **kwargs):
        return SMPLXDatastruct(self, **kwargs)

@dataclass
class SMPLXDatastruct(Datastruct):
    transforms: SMPLXTransform

    features: Optional[Tensor] = None
    pose: Optional[Tensor] = None
    rot6d: Optional[Tensor] = None
    rotmat: Optional[Tensor] = None
    rfeats: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ["features", 'rot6d', 'rotmat', 'pose']

        self.rfeats = self.features

    # features or other formats can be added later
    def __len__(self):
        return len(self.features)

def collate_data_and_text(element_list):

    collate_datastruct = element_list[0]["datastruct"].transforms.collate
    batch = {
        "datastruct": collate_datastruct([e["datastruct"] for e in element_list]),
        "text": [e["text"] for e in element_list],
        "length": [e["length"] for e in element_list]
    }

    otherkeys = [x for x in element_list[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [e[key] for e in element_list]
    return batch

class RotIdentityTransform(Transform):
    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return RotTransDatastruct(**kwargs)

    def __repr__(self):
        return "RotIdentityTransform()"


@dataclass
class RotTransDatastruct(Datastruct):
    rots: Tensor
    trans: Tensor

    transforms: RotIdentityTransform = RotIdentityTransform()

    def __post_init__(self):
        self.datakeys = ["rots", "trans"]

    def __len__(self):
        return len(self.rots)

if __name__ == "__main__":
    # test collate_tensor_with_padding
    import torch
    a = torch.randn(5, 5)
    b = torch.randn(7, 3)
    c = torch.randn(6, 4)
    batch = [a, b, c]

