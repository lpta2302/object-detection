import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from ultralytics.utils.downloads import download
import os

# Config
yaml = {
    "path": "/content/VOC",
    "names": {
        0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
        5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow", 10: "diningtable",
        11: "dog", 12: "horse", 13: "motorbike", 14: "person",
        15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor"
    }
}

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1.0 / size[0], 1.0 / size[1]
        x, y = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1
        w, h = box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f"VOC{year}/Annotations/{image_id}.xml")
    out_file = open(lb_path, "w")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    names = list(yaml["names"].values())
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls in names and int(obj.find("difficult").text) != 1:
            xmlbox = obj.find("bndbox")
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
            cls_id = names.index(cls)
            out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + "\n")

# Download only 2012 dataset
dir = Path(yaml["path"])
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_11-May-2012.zip"
download(url, dir=dir / "images", curl=True, threads=3, exist_ok=True)

# Convert only 2012 (train + val)
path = dir / "images/VOCdevkit"
for image_set in ("train", "val"):
    year = "2012"
    imgs_path = dir / "images" / f"{image_set}{year}"
    lbs_path = dir / "labels" / f"{image_set}{year}"
    imgs_path.mkdir(exist_ok=True, parents=True)
    lbs_path.mkdir(exist_ok=True, parents=True)

    with open(path / f"VOC{year}/ImageSets/Main/{image_set}.txt") as f:
        image_ids = f.read().strip().split()

    for id in tqdm(image_ids, desc=f"{image_set}{year}"):
        img_file = path / f"VOC{year}/JPEGImages/{id}.jpg"
        lb_file = (lbs_path / img_file.name).with_suffix(".txt")
        os.rename(img_file, imgs_path / img_file.name)
        convert_label(path, lb_file, year, id)