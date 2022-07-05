from config import TARGET
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(BASE, "split")
SAMPLE_FACTOR = 25

def get_all_frames():
    scans = sorted(os.listdir(TARGET))
    frames = {}
    for s in scans:
        print("checking scan:", s)
        scan_folder = os.path.join(TARGET, s)
        frames[s] = sorted(list(set([x.split(".")[0] for x in os.listdir(scan_folder) if "frame" in x])))

    print("find {:d} scans with totally {:d} frames".
          format(len(frames), sum([len(frames[key]) for key in scans])))
    return frames


def get_split(split="train"):
    assert split in ["train", "test", "val"]
    file = os.path.join(SPLIT, "scannetv2_{:s}.txt".format(split))
    with open(file, "r") as f:
        lines = f.read().splitlines()
    return lines


def sample(frames, split, factor):
    assert split in ["train", "test", "val"]    
    scans = get_split(split)
    result = []
    for s in scans:
        f = frames[s]
        result += [s+"/"+f[i] for i in range(0, len(f), factor)]
    print("sampled {:d} frames in the {:s} set".format(len(result), split))
    return result


def write(result, split, factor):
    save = "sampled_{:s}_{:d}.txt".format(split, factor)
    with open(save, "w") as f:
        for r in result:
            f.write(r+"\n")


if __name__ == "__main__":
    frames = get_all_frames()
    ret = sample(frames, "train", SAMPLE_FACTOR)
    write(ret, "train", SAMPLE_FACTOR)
    ret = sample(frames, "val", SAMPLE_FACTOR)
    write(ret, "val", SAMPLE_FACTOR)
    
