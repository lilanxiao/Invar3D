'''
used to copy sampled frames to a new folder. 
useful for transfering data across computers.
'''

import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from config import TARGET
SCANS = sorted(os.listdir(TARGET))
TRAIN_FRAMES = os.path.join(BASE_DIR, "sampled_train_25.txt")
VAL_FRAMES = os.path.join(BASE_DIR, "sampled_val_25.txt")

# ---------------------------------------------------
# the path to save, please update!
SAVE = None
# ---------------------------------------------------

def copy_images(log):
    with open(log, "r") as f:
        files = f.read().splitlines()
    for i, l in enumerate(files):
        # depth
        src = os.path.join(TARGET, l+".png")
        # rgb
        src2 = os.path.join(TARGET, l+".color.jpg")
        # pose
        pose = os.path.join(TARGET, l+".pose.txt")
        # destination
        dst = os.path.join(SAVE, l.split("/")[0])
        shutil.copy2(src, dst)
        shutil.copy2(src2, dst)
        shutil.copy2(pose, dst)
        if i%1000 == 0:
            print("... {:d} frames copied".format(i))


def main():
    # create sub folders
    for s in SCANS:
        f = os.path.join(SAVE, s)
        if not os.path.exists(f):
            os.mkdir(f)

    # copy calibrations
    for s in SCANS:
        f = os.path.join(SAVE, s)
        c = os.path.join(TARGET, s, "_info.txt")
        shutil.copy2(c, f)

    # copy depth maps and rgb images
    copy_images(TRAIN_FRAMES)
    copy_images(VAL_FRAMES)
    print("ok")


if __name__ == "__main__":
    main()
    