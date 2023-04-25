## Convert bmp image files to mask multi type for coco prepare use
#%%
from PIL import Image
import numpy as np
import os
import time
import tqdm
import argparse

'''
Here we convert the multi class mask images (shape: (H, W) , 8-bit) to 
    pycococreatortools mask (24-bit png file/ 8-bit png file).
    
Note:
    Here, input_folder is the mask folder;
    output_folder is the output annotations folder.
'''
#### ******** Modify the mask folder here
input_folder = r"图片格式数据/valid/anno/"
output_folder = r"图片格式数据/valid/temp-singlemask/"





#### -----------
#%%
def read_image_tonumpy(filename: str):
    '''Read an image in numpy format'''
    img = Image.open(filename)
    # 转换成np.ndarray格式
    img = np.array(img)
    return img

#### **** convert single mask to multi-class mask
def convert_to_multimask(mask: np.ndarray):
    '''
    Return a list of multiple masks with [0,1]
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

def save_mask_to_png(mask: np.ndarray, filename: str):
    '''Save single mask to png file.
    Args:
        mask: shape (H, W) 2d array, only contains value [0,1];
        filename : object png file name.
    '''
    mask = mask*255
    # img = np.array([mask, mask, mask, np.ones(mask.shape)*255], dtype=np.uint8)
    # img = np.transpose(img, (1,2,0))
    # We find that it seems that the file bit type does not matter.
    img = np.array(mask, dtype=np.uint8)

    # translate to Image object
    img = Image.fromarray(img)
    img.save(filename)

def save_multiclasses_to_png(multiclass, types, out_folder: str, img_name: str):
    '''
    Args:
        multiclass: contains a list of masks;
        types: contains the class with id number;
        out_folder: object folder path;
        img_name: original image file name.

    Note:
    'types' is not used yet.
    '''
    num = len(multiclass)
    for i in range(num):
        each_mask = multiclass[i]
        # each_type = types[i-1]
        __, name = os.path.split(img_name)
        name, ext = os.path.splitext(name)
        ## ** save name type
        name = name + "_tooth_" + str(i) + '.png' # imageid
        filename = os.path.join(out_folder, name)
        # save png files
        save_mask_to_png(each_mask, filename)
    # the following are used for multi class tooth process
    # for i in range(1, num+1):
    #     each_mask = multiclass[i-1]
    #     # each_type = types[i-1]
    #     __, name = os.path.split(img_name)
    #     name, ext = os.path.splitext(name)
    #     ## ** save name type
    #     # name = name + "_tooth_" + str(i-1) + '.png' # imageid
    #     name = name + "_tooth" + str(i) + '_0.png' # imageid
    #     filename = os.path.join(out_folder, name)
    #     # save png files
    #     save_mask_to_png(each_mask, filename)


#### **** Here convert single image to precoco-like png files
def convert_imagemask_to_precoco_png(filename: str, obj_path: str):
    # 读取成 mask
    img_mask = read_image_tonumpy(filename)
    # 转换成 multi-mask
    multiclass, types = convert_to_multimask(img_mask)
    # 保存成多个 mask
    save_multiclasses_to_png(multiclass, types,  obj_path, filename)


### used for batch processing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default=None)

    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    ## replace the setting
    args = parse_args()
    if args.input_folder:
        input_folder = args.input_folder
    if args.output_folder:
        output_folder = args.output_folder


    print("Preparing to process...")
    time.sleep(2)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # # convert multi-image task : input_folder --> output_folder
    filelist = os.listdir(input_folder)
    for filename in tqdm.tqdm(filelist):
        _, ext = os.path.splitext(filename)
        if str.lower(ext) in ['.jpg', '.png', '.gif']:
            filename = os.path.join(input_folder, filename)
            convert_imagemask_to_precoco_png(filename, output_folder)

    # convert_image_to_precoco_png(input_folder, output_folder)
    print("End of process !")
