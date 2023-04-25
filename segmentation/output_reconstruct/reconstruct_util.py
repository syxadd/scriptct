import numpy as np
import os                #遍历文件夹
import nibabel as nib    #nii格式一般都会用到这个包
from PIL import Image
from typing import List
import re

def read_image_tonumpy(filename: str):
    '''Read an image in numpy format'''
    img = Image.open(filename)
    # 转换成np.ndarray格式
    img_array = np.array(img)
    img.close()
    return img_array	

def _get_3darray_fromlist(filelist: List[str]):
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

def _get_all_names(folder: str):
	'''Get Image Files with number id in an folder.
	Does not contain path, only name. 
	注：测试时每次只能测试一个实例
	'''
	namelist = [name for name in os.listdir(folder)]
	if len(namelist) == 0:
		return namelist
	# to contruct re expression 
	rexp = re.compile(r"\d+")
	# sort by number
	namelist.sort(key=lambda s: int(re.findall(rexp, s)[0]) )
	return namelist

def get_3darray_fromfolder(folder: str):
	"""从一个实例的二维图片获得所有层的图重建3维数组。
	folder为输入的文件夹，包含了所有的mask。每个文件夹只能有一个个体。
	"""
	namelist = _get_all_names(folder)
	filelist = [os.path.join(folder, name) for name in namelist]
	data_array = _get_all_names(filelist)
	# print(data_array.shape)
	data_array = np.transpose(data_array, [1,2,0])
	# print(data_array.shape)
	return data_array

if __name__ '__main__':
	_, name  = os.path.split(folder)

