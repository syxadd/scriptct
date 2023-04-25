#%%
from PIL import Image
import numpy as np
import os
import time
from tqdm import tqdm
import shutil

'''
将多分类的mask导出二分类（1-分类 和 0-背景）的标注。一次处理一个文件夹。
图像仍然使用原有的图像，不需要做处理。

使用：修改 origin_mask_folder 为多标注文件夹路径，output_mask_folder 为输出mask路径。
图片路径可复制可不复制，不复制就写None。

Here we convert the multi class mask images (shape: (H, W) , 8-bit) to 
    only single semantic mask (8-bit png file).
    
Note:
    Here, input_folder is the mask folder;
    output_folder is the output single mask folder.
'''

#### Options
#### ******** Modify the mask folder here
origin_image_folder = "train/images"
output_image_folder = "unet-data/images"

origin_mask_folder = "train/masks"
output_mask_folder = "unet-data/masks"





#%% ---- Run
def read_image_tonumpy(filename: str):
    '''Read an image in numpy format'''
    img = Image.open(filename)
    # 转换成np.ndarray格式
    img = np.array(img)
    return img

def convert_mask(mask: np.ndarray):
    out = (mask > 0) * 1
    return out

def write_mask(origin_path: str, obj_path: str):
    mask = read_image_tonumpy(origin_path)
    mask = convert_mask(mask)
    img = Image.fromarray(np.uint8(mask))
    img.save(obj_path)


def copy_allfiles(path: str, obj_path: str, mode='data'):
    """Copy all files in a folder to the path.  
    """
    namelist = os.listdir(path)
    for name in namelist:
        if mode == 'data':
            shutil.copy(os.path.join(path, name), os.path.join(obj_path, name))
        elif mode == 'mask':
            write_mask(os.path.join(path, name), os.path.join(obj_path, name))
        else:
            raise ValueError("mode is wrong: "+ mode)



if __name__ == '__main__':
    print("Preparing to process...")
    time.sleep(2)

    # copy images 
    if output_image_folder is not None and len(output_image_folder) > 0:
        if not os.path.exists(output_image_folder):
            os.mkdir(output_image_folder)
        print("Copy images to folder:", output_image_folder)

        folder_list = os.listdir(origin_image_folder)
        for folder in tqdm(folder_list):
            folder = os.path.join(origin_image_folder, folder)
            if os.path.isdir(folder):
                copy_allfiles(folder, output_image_folder, mode="data")

    print("End of copying images.")

    # process and save masks
    if output_mask_folder is not None and len(output_mask_folder) > 0:
        if not os.path.exists(output_mask_folder):
            os.mkdir(output_mask_folder)
        print("Copy masks to folder : ", output_mask_folder)

        folder_list = os.listdir(origin_mask_folder)
        for folder in tqdm(folder_list):
            folder = os.path.join(origin_mask_folder, folder)
            if os.path.isdir(folder):
                copy_allfiles(folder, output_mask_folder, mode="mask")

    print("End of process masks.")


    print("End of process !")