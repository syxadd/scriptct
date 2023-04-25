#!/usr/bin/env python3
import datetime
import json
import os
import re
import argparse
import fnmatch
import numpy as np
from PIL import Image
from itertools import groupby
from pycococreatortools import pycococreatortools
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from tqdm import tqdm


'''
使用方法：
python toothdata_to_coco2022.py --root_dir 图片格式数据/train/ \
 --out_filename "instances_tooth_train2021.json"
'''


## --------------------
## Here to modifty the directory
# os.chdir(r"D:\user-syxpop\Documents\using-CVgroup\CTsegmentationTask\data-to-use\prepare-for-cocodata\tooth")

ROOT_DIR = ''
IMAGE_DIR = os.path.join(ROOT_DIR, "CT")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "temp-singlemask")

OUTNAME = "instances_tooth_valid2021"

# debug_status = False
# if debug_status:
#     ROOT_DIR = 'train-example'
#     IMAGE_DIR = os.path.join(ROOT_DIR, "tooth_image")
#     ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")








## --------------------
## Here are the infos
INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'tooth',
        'supercategory': 'tooth',
    },
]

# CATEGORIES = []
# for i in range(1, 17):
#     CATEGORIES.append({
#         'id': i, 'name': 'tooth%d' % i, 'supercategory': 'tooth',
#     })

# CATEGORIES = [
#     {
#         'id': 1,
#         'name': 'tooth',
#         'supercategory': 'tooth',
#     },
#     # {
#     #     'id': 3,
#     #     'name': 'triangle',
#     #     'supercategory': 'shape',
#     # },
# ]



## --------------------

## utils
def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.bmp', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files

def filter_for_tooth_annotations(root, files, image_filename):
    # modified from tooth_to_coco.py
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_tooth_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    # with open('{}/{}.json'.format(ROOT_DIR, OUTNAME), 'w') as output_json_file:
    #     json.dump(coco_output, output_json_file)
    if OUTNAME.endswith(".json"):
        out_filename = os.path.join(ROOT_DIR, OUTNAME)
    else:
        out_filename = os.path.join(ROOT_DIR, '{}.json'.format(OUTNAME) )
    with open(out_filename, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)




### used for batch processing
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default=None)
    # parser.add_argument("--annotation_folder", type=str, default=None)
    # args.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--out_filename", type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # replace infos
    args = parse_args()
    if args.root_dir:
        ROOT_DIR = args.root_dir
    if args.out_filename:
        OUTNAME = args.out_filename
    IMAGE_DIR = os.path.join(ROOT_DIR, "CT")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "temp-singlemask")

    # running scripts
    main()
