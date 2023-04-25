import os
import numpy as np
import nibabel as nib
import argparse
from PIL import Image
from tqdm import tqdm

'''
检查所有图的shape是否一致
'''
### 
image_folder = "train/CT/"
mask_folder = "train/anno/"


#### ---- 处理
def read_nii_to_3darray(nii_path: str, with_shape="WHC", mode="data" ):
    '''
    Read nii file to 3d numpy array with shape (x, y, z), default return shape is (x, y, z)
    '''
    nii_img = nib.load(nii_path)
    img_array = nii_img.get_fdata()

    ## with preproces
    if mode=='data':
        # img_array = _image3D_normalize(img_array)
        pass
    elif mode=='mask':
        pass
    else:
        raise ValueError("mode "+mode+" not defined.")

    # adjust return shape
    if with_shape == "WHC":
        return img_array
    elif with_shape == "CHW":
        return img_array.transpose([2,1,0])
    else:
        raise ValueError("type " + str(with_shape) + "is wrong.")

def main():
    namelist = os.listdir(image_folder)
    for name in namelist:
        image_name = os.path.join(image_folder, name)
        mask_name = os.path.join(mask_folder, name)
        image_shape = read_nii_to_3darray(image_name).shape
        mask_shape = read_nii_to_3darray(image_name).shape
        print("image name:", name, "shape:", image_shape)
        print("mask name:", name, "shape:", mask_shape)

    print("End.")

if __name__ == '__main__':
    main()