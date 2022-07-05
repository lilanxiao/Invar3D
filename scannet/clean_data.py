import numpy as np
from torch.utils.data import DataLoader
from scannet_pretrain import ScanNetContrastBase
from sampler import get_all_frames, sample

class Wrapper(ScanNetContrastBase):
    
    def __init__(self, split: str, frames=None) -> None:
        super(Wrapper, self).__init__(split, False, frames)
    
    def __getitem__(self, index):
        path = self.frames[index]
        scan, f = path.split("/")
        depth, intr =  self._get_one_frame(scan, f)
        # find depth maps with all zeros
        if np.mean(depth) < 1e-6:
            print(path)
        return depth, intr


def all_in_split(split="train"):
    frames = get_all_frames()
    return sample(frames, split, 1)


def find_invalid(split="train"):
    frames = all_in_split(split)
    ds = Wrapper(split, frames)
    # use multiprocessing for speed
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6)
    for data in dl:
        pass


if __name__ == "__main__":
    find_invalid("train")
    