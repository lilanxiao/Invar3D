import os
import cv2
import time

# ------------------- put your path here ---------------------
SOURCE = ""     # where the raw data is save
TARGET = ""     # where you what to put the extracted data
# ------------------------------------------------------------
SCANS = sorted(os.listdir(SOURCE))

def extract_one_scan(scan_name):
    source = os.path.join(SOURCE, scan_name, scan_name + ".sens")
    target = os.path.join(TARGET, scan_name)
    if not os.path.exists(target):
        os.mkdir(target)

    """
    # python, slow
    os.system(
        "python read.py --filename {} --output_path {} --export_depth_images --export_poses --export_intrinsics"
        .format(source, target)
    )
    """
    # c++
    os.system(
        "./sen {} {}".format(source, target)
    )


def remove_jpg(scan_name):
    """RGB images take too much space. consider to remove them

    Args:
        scan_name ([str]): name of scan
    """
    print("remove color images ...")
    target = os.path.join(TARGET, scan_name)
    data_list = os.listdir(target)
    for data in data_list:
        if ".color.jpg" in data:
            os.remove(os.path.join(target, data))


def resize_jpg(scan_name):
    """RGB images have higher resolutions than depth maps.
    Resize them to save space

    Args:
        scan_name ([str]): name of scan
    """
    print("resize color images to the same shape of depth maps ...")
    target = os.path.join(TARGET, scan_name)
    data_list = os.listdir(target)
    for data in data_list:
        if ".color.jpg" in data:
            frame_name = data.split(".")[0]
            dpth = cv2.imread(os.path.join(target, frame_name + ".png"), cv2.IMREAD_UNCHANGED)
            color = cv2.imread(os.path.join(target, data))
            if dpth is None:
                print("no depth")
            elif color is None:
                print("no rgb")
            else:
                h, w = dpth.shape[0], dpth.shape[1]
                color = cv2.resize(color, (w, h), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(target, data), color)


def pgm2png(scan_name):
    print("convert pgm to png ...")
    target = os.path.join(TARGET, scan_name)
    data_list = os.listdir(target)
    for data in data_list:
        if ".depth.pgm" in data:
            name = data.split(".depth")[0]
            img = cv2.imread(os.path.join(target, data), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(target, name+".png"), img)
            os.remove(os.path.join(target, data))


def unpack_one_scan(scan_name):
    start = time.time()
    extract_one_scan(scan_name)
    pgm2png(scan_name)
    resize_jpg(scan_name)           # resize rgb images to save space
    # remove_jpg(scan_name)         # remove rgb images to save space
    t = time.time() - start
    print(".... scan {} unpacked!, time: {:.3} s ....".format(scan_name, t))


if __name__ == "__main__":
    for s in SCANS:
        unpack_one_scan(s)
