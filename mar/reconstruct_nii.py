import numpy as np
import os                #遍历文件夹
import nibabel as nib    #nii格式一般都会用到这个包
from PIL import Image
import cv2
import h5py
from typing import List
import re
'''
**MAR usage**
将二维图像序列重建为3D nii数组

说明：
    输入input_folder 为二维图像序列文件夹；
    输出目录output_folder指定了输出文件夹，配合output_name指定输出的文件名，确定保存的文件路径；
    refer_file为每个重建的数组对应的原CT文件，主要提取其中的位置信息来构建标注；
    head 为头部插入空白层数，tail 为尾部插入空白层数，为0就表示不插入序列。最终要保证结果的层数与ct的层数一致才能打开。
'''
## ---- ****
# 注：在此修改相关参数。
input_folder = r"outputH5\output-data3"
output_folder = r"outputH5"
output_name = 'pred-data3.nii.gz'

refer_file = r"data2.nii.gz"
# head = 408
# tail = 626-569

## 获得读入的图片所在原CT的位置，其他位置插入0序列补全，这个通常用于补充分割mask的结果
slice_start = 0
slice_end = 608

# 整个nii文件的slice数量
total_length = 699
# mode 处理模式：如果是png图片用"image", 如果是h5文件用"h5"
mode = "image"
## ----




## -----------------
def read_image_tonumpy(filename: str):
    """Read an image in numpy format"""
    img = Image.open(filename)
    # 转换成np.ndarray格式
    img_array = np.array(img)
    img.close()
    return img_array

def read_niifile(nii_filepath: str):
    # return 3D image
    img_nii = nib.load(nii_filepath)  # 读取nii
    return img_nii

def get_all_files(folder: str):
    """Get Image Files with number id in an folder.
    Does not contain path, only name.
    注：测试时每次只能测试一个实例，因为文件名特别针对数字进行了排序，不能使用多个实例。
    """
    namelist = [name for name in os.listdir(folder)]
    if len(namelist) == 0:
        return namelist
    # to contruct re expression
    rexp = re.compile(r"\d+")
    # sort by number
    namelist.sort(key=lambda s: int(re.findall(rexp, s)[0]) )
    return namelist

def get_3darray(filelist: List[str]):
    # filelist = [name for name in os.listdir(folder)]
    # filelist = [os.path.join(folder, name) for name in os.listdir(folder)]
    datalist = []
    for filename in filelist:
        img_array = read_image_tonumpy(filename)
        ## 镜像翻转，矩阵需要进行转置复原
        img_array = img_array.T
        datalist.append(img_array)

    data = np.asarray(datalist)
    return data

def insert_blankarray(data_array: np.ndarray, head: int, tail: int):
    """
    在3d数组前后插入一定数量的序列。head为头部插入的层数（数量>=0），tail为尾部插入的层数（数量>=0）。
    """
    z,y,x = data_array.shape
    if head > 0:
        head_part = np.zeros((head, y, x), dtype=data_array.dtype)
        data_array = np.concatenate([head_part, data_array], axis=0)
    if tail > 0:
        tail_part = np.zeros((tail, y, x), dtype=data_array.dtype)
        data_array = np.concatenate([data_array, tail_part], axis=0)

    return data_array

def save_to_nii(data_array: np.ndarray, filename: str, affine_array: np.ndarray):
    ## refer_data is nii_img
    data_array = data_array.astype(np.float)
    nii_img = nib.Nifti1Image(data_array, affine_array)
    nib.save(nii_img, filename)






############ For MAR usage

def read_exported_datah5_slices(filename: str, normalize=False, window_size = (None, None), mask_threshold: float or None = None):
    '''
    Use for test process. `window_size` is the tuple with (minHU, maxHU).
    Currently do not truncate values.
    '''
    with h5py.File(filename, mode='r') as f:
        data_ma = np.array(f['image']) # read with shape (H, W)

        if mask_threshold is None:
            data_metalmask = np.array(f['metal_mask']) # read with shape (H, W)
        else:
            data_metalmask = data_ma > mask_threshold

    if normalize and window_size[1] is not None:
        minval, maxval = window_size
        # minval = minval / 1000 * miuWater + miuWater
        # maxval = maxval / 1000 * miuWater + miuWater

        data_ma = (data_ma - minval) / (maxval - minval)


    # return all data with single image shape (H, W)
    data = {
        "filename": filename,
        # "gt_image": data_gt,
        "ma_image": data_ma,
        "metal_mask": data_metalmask,

    }
    return data

def get_3darray_fromH5file(filelist: List[str]):
    """
    Read image array from h5 files.
    """
    # filelist = [name for name in os.listdir(folder)]
    # filelist = [os.path.join(folder, name) for name in os.listdir(folder)]
    datalist = []
    for filename in filelist:
        img_array = read_exported_datah5_slices(filename, normalize=False)['ma_image']
        ## 镜像翻转，矩阵需要进行转置复原
        img_array = img_array.T
        # img_array = img_array.astype(float)
        # img_array = cv2.resize(img_array, dsize=(obj_size, obj_size), interpolation=cv2.INTER_LINEAR)
        datalist.append(img_array)

    data = np.asarray(datalist)
    # data = np.int32(data.round())
    return data







def main():
    # filelist = ['GAO JIA1', 'GAO JIA2', 'GAO JIA10', 'GAO JIA12', 'GAO JIA15']
    namelist = get_all_files(input_folder)
    filelist = [os.path.join(input_folder, name) for name in namelist]
    print("Processing folder: ", os.path.split(input_folder)[1])
    # data_array = get_3darray(filelist)
    if mode == "image":
        data_array = get_3darray(filelist)
    elif mode == "h5":
        data_array = get_3darray_fromH5file(filelist)
    else:
        data_array = get_3darray(filelist)

    ## insert blank part
    head_num = slice_start - 0
    tail_num = total_length - 1 - slice_end
    if head_num > 0 or tail_num > 0:
        data_array = insert_blankarray(data_array, head_num, tail_num)

    print("处理完的层数一共有：",data_array.shape[0])
    # transpose the shape
    data_array = np.transpose(data_array, [1,2,0])
    print("处理完的数组形状为：", data_array.shape)

    ## save to nii_file
    refer_data = nib.load(refer_file)
    save_to_nii(data_array, filename=os.path.join(output_folder, output_name), affine_array=refer_data.affine )

    print("End of the process. ")



if __name__ == '__main__':
    main()


