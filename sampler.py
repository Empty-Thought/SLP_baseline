from typing import Optional
from dataclasses import dataclass
import numpy as np
from numpy import ndarray as Array
import random

def get_frameix_from_data_index(num_frames: int,
                                request_frames: Optional[int],
                                sampling: str = "conseq",
                                sampling_step: int = 1) -> Array:
    nframes = num_frames

    if request_frames is None:
        frame_ix = np.arange(nframes)
    else:
        # sampling goal: input: ----------- 11 nframes
        #                       o--o--o--o- 4  ninputs
        #
        # step number is computed like that: [(11-1)/(4-1)] = 3
        #                   [---][---][---][-
        # So step = 3, and we take 0 to step*ninputs+1 with steps
        #                   [o--][o--][o--][o-]
        # then we can randomly shift the vector
        #                   -[o--][o--][o--]o
        # If there are too much frames required
        if request_frames > nframes:
            fair = False  # True
            if fair:
                # distills redundancy everywhere
                choices = np.random.choice(range(nframes),
                                           request_frames,
                                           replace=True)
                frame_ix = sorted(choices)
            else:
                # adding the last frame until done
                ntoadd = max(0, request_frames - nframes)
                lastframe = nframes - 1
                padding = lastframe * np.ones(ntoadd, dtype=int)
                frame_ix = np.concatenate((np.arange(0, nframes),
                                           padding))

        elif sampling in ["conseq", "random_conseq"]:
            step_max = (nframes - 1) // (request_frames - 1)
            if sampling == "conseq":
                if sampling_step == -1 or sampling_step * (request_frames - 1) >= nframes:
                    step = step_max
                else:
                    step = sampling_step
            elif sampling == "random_conseq":
                step = random.randint(1, step_max)

            lastone = step * (request_frames - 1)
            shift_max = nframes - lastone - 1
            shift = random.randint(0, max(0, shift_max - 1))
            frame_ix = shift + np.arange(0, lastone + 1, step)

        elif sampling == "random":
            choices = np.random.choice(range(nframes),
                                       request_frames,
                                       replace=False)
            frame_ix = sorted(choices)

        else:
            raise ValueError("Sampling not recognized.")

    return frame_ix

@dataclass
class FrameSampler:
    sampling: str = "conseq"
    sampling_step: int = 1
    request_frames: Optional[int] = None
    threshold_reject: int = 0.75
    max_len: int = 1000
    min_len: int = 10

    def __call__(self, num_frames):
        return get_frameix_from_data_index(num_frames,
                                           self.request_frames,
                                           self.sampling,
                                           self.sampling_step)

    def accept(self, duration):
        # Outputs have original lengths
        # Check if it is too long
        if self.request_frames is None:
            if duration > self.max_len:
                return False
            elif duration < self.min_len:
                return False
        else:
            # Reject sample if the length is
            # too little relative to
            # the request frames
            min_number = self.threshold_reject * self.request_frames
            if duration < min_number:
                return False
        return True

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


