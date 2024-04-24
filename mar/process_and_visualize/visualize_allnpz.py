from typing_extensions import dataclass_transform
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import h5py

'''
Read npz files and output png files to the OUT_FOLDER.
'''

## with ma
IN_FOLDER = "D:/ShenData/output_datasets/export_slices/data_withgt/ma/1001932549-20170707"
OUT_FOLDER = "D:/ShenData/output_datasets/export_slices/visual-ma-1001932549-20170707"


## no ma
# IN_FOLDER = "D:/ShenData/output_datasets/export_slices/data_withgt/noma/1002206265-20171007"
# OUT_FOLDER = "D:/ShenData/output_datasets/export_slices/visual-noma-1002206265-20171007"




def visualize_file_save(in_file: str, out_file: str):
	data = np.load(in_file)
	img = data['image']
	mask = data['mask']

	LOW = -1000
	HIGH = 2500

	img = adjust_ct_window(img, minval=LOW, maxval=HIGH)
	img = (img - LOW) / (HIGH - LOW) * 255
	skio.imsave(out_file, np.uint8(img))

def visualize_h5file_save(in_file: str, out_file: str):
	with h5py.File(in_file, 'r') as f:
		img = f['image']
		mask = f['mask']
		metal_mask = f['metal_mask']

	LOW = -1000
	HIGH = 2500

	img = adjust_ct_window(img, minval=LOW, maxval=HIGH)
	img = (img - LOW) / (HIGH - LOW) * 255
	skio.imsave(out_file, np.uint8(img))

def visualize_h5file_output_save(in_file: str, out_file: str):
	"""
	visualize the matlab output files. (generate ma)
	"""
	with h5py.File(in_file, 'r') as f:
		data_ct = np.asarray(f['gt_CT']).T
		data_ma = np.asarray(f['ma_CT'])[0].T
		metal_mask = np.asarray(f['metal_mask'])[0].T

	LOW = np.min(data_ct)
	HIGH = np.max(data_ct)

	data_ct = (data_ct - LOW) / (HIGH - LOW) * 255
	data_ma = adjust_ct_window(data_ma, minval=LOW, maxval=HIGH)
	data_ma = (data_ma - LOW) / (HIGH - LOW) * 255

	filepre, ext = os.path.splitext(out_file)
	out_ma_file = filepre+'_ma'+ext
	out_mask_file = filepre + '_mask' + ext
	skio.imsave(out_file, np.uint8(data_ct))
	skio.imsave(out_ma_file, np.uint8(data_ma))
	skio.imsave(out_mask_file, np.uint8(metal_mask))

def adjust_ct_window(imgdata: np.ndarray, minval, maxval, inplace=True):
	if inplace:
		data = imgdata
	else:
		data = imgdata.copy()
	data[data < minval] = minval
	data[data > maxval] = maxval

	return data

if __name__ == "__main__":
	namelist = os.listdir(IN_FOLDER)
	if not os.path.exists(OUT_FOLDER):
		os.makedirs(OUT_FOLDER)
	for name in namelist:
		if not name.endswith(".npz"):
			continue
		in_file = os.path.join(IN_FOLDER, name)
		out_file = os.path.join(OUT_FOLDER, name.replace(".npz", ".png"))

		visualize_file_save(in_file=in_file, out_file=out_file)
		print("Draw the img:", name)

	print("End of the process.")