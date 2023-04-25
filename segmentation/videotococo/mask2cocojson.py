from sys import getallocatedblocks
import numpy as np
import pycocotools.mask as cocomask
import os
# import glob
import json

from PIL import Image
from itertools import groupby
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from tqdm import tqdm

import argparse
import re
# import using_debug
#### --------------- 说明
'''
上一步操作：将文件夹里的所有nii文件全部进行分解，成多个小的文件，然后分成多个文件夹。
在完成上一步操作之后，将导出的分片文件夹作为video来处理，这里将其处理为vis任务用的json格式。
'''

image_folder = r"valid-debug/images/"
mask_folder = r"valid-debug/masks/"


output_folder = r"valid-debug"
out_filename = "anno_valid-debug.json"







# Define which colors match which categories in the images
# category_colors = {
#     "(0, 0, 0)": 0, # Outlier
#     "(255, 0, 0)": 1, # Window
#     "(255, 255, 0)": 2, # Wall
#     "(128, 0, 255)": 3, # Balcony
#     "(255, 128, 0)": 4, # Door
#     "(0, 0, 255)": 5, # Roof
#     "(128, 255, 255)": 6, # Sky
#     "(0, 255, 0)": 7, # Shop
#     "(128, 128, 128)": 8 # Chimney
# }
out_filename = os.path.join(output_folder, out_filename)


#### --------------- 处理
## some operations
## io
def read_image_tonumpy(filename: str):
    '''Read an image in numpy format'''
    img = Image.open(filename)
    # 转换成np.ndarray格式
    img = np.array(img)
    return img

def convert_to_multimask(mask: np.ndarray):
    '''
    Return a list of multiple masks with [0,1] values. 
    Args:
        mask: shape (H, W) 2d array with some kind range [0, 255]
            for example: [[0,2,3,5,6,...]]
    Return:
        multiclasses: a list that each elements is a mask with values [0,1]
        types: a list that saves all class values in mask except 0 .
    '''
    types = np.unique(mask)
    multiclasses = []
    names = []

    for i in types:
        # omit 0 mask value
        if i == 0:
            continue
        each_mask = np.zeros(mask.shape)
        each_mask[mask == i] = 1
        multiclasses.append(each_mask)
        names.append(i)
    
    types = names
    return multiclasses, types

def binary_mask_to_rle(binary_mask):
    """
    binary mask to rle, binary_mask has value: 0,1
    return a dict with a list of rle values and the size of the mask;
    In the list, even index (starts from 0) is 0's number, odd index (starts from 1) is 1's number

    Return:
        {
            'counts': list,
            'size': (Height, Width)
        }
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images

def get_maximum_bbox(mask: np.ndarray):
    """
    Find the maximum bounding box.
    Return:
        (top, left, bottom, right)
    """
    skmask = sk_label(mask)
    regions = sk_regions(skmask)
    shape = mask.shape
    xmin = shape[1]
    ymin = shape[0]
    ymax, xmax = 0, 0
    for region in regions:
        top, left, bottom, right = region.bbox
        if top < ymin:
            ymin = top
        if left < xmin:
            xmin = left
        if bottom > ymax:
            ymax = bottom
        if right > xmax:
            xmax = right

    return ymin, xmin, ymax, xmax

def get_all_files(folder: str):
	'''Get Image Files with number id in an folder.
	Does not contain path, only name. 
	注：测试时每次只能测试一个实例
	'''
	namelist = os.listdir(folder)
	if len(namelist) == 0:
		return namelist
	# to contruct re expression 
	rexp = re.compile(r"\d+")
	# sort by number
	namelist.sort(key=lambda s: int(re.findall(rexp, s)[0]) )
	return namelist
## code process
## transform process
def _annotation_info(image_folder: str, mask_folder: str, all_types: int):
    '''
    获得每一个video的annotation list。处理的annotation_list 为一列list，直接成为了独立的annotations，其中id需要修改，videoid也需要对应修改。
    all_types 为每个标注中实例的个数，可以设置为32；
    返回一个annotation_list。
    '''
    # 找到有标号次序的namelist
    namelist = get_all_files(image_folder)
    assert len(namelist) > 0, 'image_folder: %s, does not have elements' % image_folder
    masks = []

    # get image mask
    for name in namelist:
        img_file = os.path.join(image_folder, name)
        mask_file = os.path.join(mask_folder, name)

        mask = read_image_tonumpy(mask_file)
        masks.append(mask)

    shape = masks[0].shape
    length = len(masks)
    # process each mask to annotations
    annotation_list = [
        {
        "id": idx, ## need to be modified after return
        "height": shape[0],
        "width": shape[1], 
        "category_id": 1, # Here also need to be modified, but we only need 1 category
        "segmentations": [None for i in range(length)],  # a list of dict
        "bboxes":[None for i in range(length)], # a list of list [x1,y1,x2,y2]
        "video_id": 0,  ## need to be modified after return, here ==0 means not processed, == 1 means has value
        "iscrowd": 0,
        "length": 1,
        "areas": [None for i in range(length)],
        } for idx in range(all_types+1)  # 设置长度+1个，用来和下边的index作映射，之后再删掉第一个
    ]

    ## classify each elements in each mask of the list masks
    for i in range(length):
        mask = masks[i]
        each_uniques = np.unique(mask)
        for value in each_uniques:
            if value == 0:
                continue
            binarymask = (mask == value)*1
            area = np.sum(binarymask).item()  ## 这里的使int32类型，需要用item转换为内置int类型

            segmentation = binary_mask_to_rle(binarymask)

            # 合并到构造的annotation_list 里面
            annotation_list[value]["segmentations"][i] = segmentation
            annotation_list[value]["areas"][i] = area
            # 增加bbox
            annotation_list[value]["bboxes"][i] = get_maximum_bbox(mask)
            annotation_list[value]["video_id"] = 1


    ## 清理多余的annotation list
    # for j in range(len(annotation_list)):
    #     if annotation_list[j]["video_id"] == 0:
    #         annotation_list.pop(j)
    # j = 0
    # N = len(annotation_list)
    # while j < N:
    #     if annotation_list[j]["video_id"] == 0:
    #         annotation_list.pop(j)
    #         N -= 1
    #     else:
    #         j += 1
    newlist = []
    for j in range(len(annotation_list)):
        if annotation_list[j]["video_id"] != 0:
            newlist.append(annotation_list[j])
    annotation_list = newlist
    
    return annotation_list



## main() 
def get_image_annotations_info(image_folder: str, mask_folder: str, out_file: str):
    '''Transform the mask to rle format in coco json.
    Args:
        image_folder (str) : Input video image folder, each video is a folder that contains multiple images;
        mask_folder (str) : Contains mask folder that corresponds to video folder;
        out_file (str) : Output filename.
    '''
    annotation_id = 1
    video_id = 1
    annotations = []
    videos = []

    folderlist = os.listdir(image_folder)

    for video_name in tqdm(folderlist):
        # 处理每个图片实例下边每张图
        each_video_folder = os.path.join(image_folder, video_name)
        namelist = get_all_files(each_video_folder)
        img_files = [video_name + '/' + name for name in namelist]
        # 处理每个video的信息
        height, width = 641, 641
        video_info = {
            "width": width,
            "height": height,
            "length": len(namelist),
            "id": video_id,
            "file_names": img_files
        }

        # 处理mask
        each_mask_folder = os.path.join(mask_folder, video_name)

        annotation_list = _annotation_info(each_video_folder, each_mask_folder, all_types=32)
        for j in range(len(annotation_list)):
            annotation_list[j]["id"] = annotation_id
            annotation_id += 1

            annotation_list[j]["video_id"] = video_id

        ## 最后video_id要+1
        video_id += 1
        videos.append(video_info)
        annotations.extend(annotation_list)


    categories = [{
        "supercategory": "organ",
        "id": 1,
        "name": "tooth"
    }]


    ## 构建json标注文件

    coco_format = {
        "info": {
            "description": "tooth ct segmentation",
            "time": "2022-01-04"
        },
        "videos": videos,
        "categories": categories,
        "annotations": annotations
    }

    ## save json style file
    with open(out_file, 'w') as f:
        json.dump(coco_format, f)

    print("End of the process.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--mask_folder", type=str, default=None)
    # args.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--out_filename", type=str, default=None)

    args = parser.parse_args()
    return args
    
    

if __name__ == "__main__":
    args = parse_args()
    if args.image_folder:
        image_folder = args.image_folder
    if args.mask_folder:
        mask_folder = args.mask_folder
    if args.out_filename:
        out_filename = args.out_filename

    print("args.image_folder: ", image_folder)
    print("args.mask_folder:", mask_folder)
    print("args.out_filename", out_filename)
    get_image_annotations_info(image_folder, mask_folder, out_filename)




    
