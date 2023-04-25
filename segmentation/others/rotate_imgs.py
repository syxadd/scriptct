# import numpy as np
from PIL import Image
import os

#### ---- Instruction
'''
将文件夹里的图像进行水平翻转。
'''
input_folder = r"结果保存\unet-predict-masks211210"
# output_folder = r''







#### ---- code part

def flip_each_file(filename: str):
	img = Image.open(filename)
	img = img.transpose(Image.FLIP_LEFT_RIGHT)
	img.save(filename)

if __name__ == '__main__':
	# filelist = [os.path.join(input_folder, name) for name in os.listdir(input_folder)]
	namelist = os.listdir(input_folder)
	for name in namelist:
		filename = os.path.join(input_folder, name)
		flip_each_file(filename)


