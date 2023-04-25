import numpy as np
import os                #遍历文件夹
import nibabel as nib    #nii格式一般都会用到这个包
from PIL import Image
from typing import List
import re
'''
将二维图像序列重建为3D nii数组

说明：
    输入input_folder 为二维图像序列文件夹；
    输出目录output_folder指定了输出文件夹，配合output_name指定输出的文件名，确定保存的文件路径；
    refer_file为每个重建的数组对应的原CT文件，主要提取其中的位置信息来构建标注；
    head 为头部插入空白层数，tail 为尾部插入空白层数，为0就表示不插入序列。最终要保证结果的层数与ct的层数一致才能打开。
'''
## ---- ****
# 注：在此修改相关参数。
input_folder = r"结果保存\output-maskrcnn-220207"
output_folder = r"结果保存\output-maskrcnn-nii"
output_name = 'maskrcnn-seg.nii.gz'

refer_file = r"CT\test.nii.gz"
head = 81
tail = 608-510
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

    data = np.array(datalist)
    return data

def insert_blankarray(data_array: np.ndarray, head: int, tail: int):
    """
    在3d数组前后插入一定数量的序列。head为头部插入的层数，tail为尾部插入的层数。
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


def main():
    # filelist = ['GAO JIA1', 'GAO JIA2', 'GAO JIA10', 'GAO JIA12', 'GAO JIA15']
    namelist = get_all_files(input_folder)
    filelist = [os.path.join(input_folder, name) for name in namelist]
    print("Processing folder: ", os.path.split(input_folder)[1])
    data_array = get_3darray(filelist)

    ## insert blank part
    if head > 0 or tail > 0:
        data_array = insert_blankarray(data_array, head, tail)

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


