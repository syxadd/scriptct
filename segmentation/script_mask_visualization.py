import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
from PIL import Image


'''
visualize the exported mask png file. 
'''

folder = r"data-to-use"
out_folder = r"data-visual"





#### --------
def get_all_files(folder: str, with_folder: bool = False):
	if with_folder:
		filelist = [os.path.join(folder, name) for name in os.listdir(folder)]
	else:
		filelist = os.listdir(folder)
	return filelist
#### read image to numpy
def read_image_tonumpy(filename: str):
    '''Read an image in numpy format'''
    img = Image.open(filename)
    # 转换成np.ndarray格式
    img_array = np.array(img)
    img.close()
    return img_array


def analyze_image_mask(filename: str):
	# img = convert_tool.read_image_tonumpy(filename)
	img = Image.open(filename)

	scale = 0.5
	w, h = img.size
	newW, newH = int(scale*w), int(scale * h)
	small_img = img.resize((newW, newH))

	img = np.asarray(img)
	small_img = np.asarray(small_img)

	dtype = img.dtype
	shape = img.shape
	uniques = np.unique(img)

	_, name = os.path.split(filename)
	print(name+" : ", "mask dtype: ", dtype, "mask shape: ", shape)
	print("\t\t  mask values: ", uniques, "--total class: ", len(uniques))
	small_uniques = np.unique(small_img)
	print("\t\t  Small mask dtype : ", small_img.dtype, "  Small mask values: ", small_uniques, "  total length: ", len(small_uniques))

def save_matplot_image(filename: str, outname: str):
	img_array = read_image_tonumpy(filename)
	plt.axis('off')
	plt.imshow(img_array)
	plt.colorbar()
	plt.savefig(outname)
	plt.close()




if __name__ == '__main__':

	allfiles = get_all_files(folder, with_folder=False)

	for filename in tqdm.tqdm(allfiles, desc="Processing: "):
		## ignore folders
		if os.path.isdir(filename):
			continue
		## show the process
		save_matplot_image(os.path.join(folder, filename), os.path.join(out_folder, filename))

