import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import h5py

'''
Read npz files and output png files to the OUT_FOLDER.
'''

## with ma
IN_FOLDER = r"D:\ShenData\output_datasets\export_slices\database_MAR-220404id2\genma\1001902453-20180113"
OUT_FOLDER = r"D:\ShenData\output_datasets\export_slices\visualize_outputh5\database_MARid2-220404\genma\1001902453-20180113"


## no ma
# IN_FOLDER = "D:/ShenData/output_datasets/export_slices/data_withgt/noma/1002206265-20171007"
# OUT_FOLDER = "D:/ShenData/output_datasets/export_slices/visual-noma-1002206265-20171007"

LOW, HIGH = -1000, 2800


def visualize_file_save(in_file: str, out_file: str):
	data = np.load(in_file)
	img = data['image']
	mask = data['mask']

	img = adjust_ct_window(img, minval=LOW, maxval=HIGH)
	img = (img - LOW) / (HIGH - LOW) * 255
	skio.imsave(out_file, np.uint8(img))

def visualize_h5file_save(in_file: str, out_file: str):
	with h5py.File(in_file, 'r') as f:
		img = f['image']
		mask = f['mask']
		metal_mask = f['metal_mask']

	img = adjust_ct_window(img, minval=LOW, maxval=HIGH)
	img = (img - LOW) / (HIGH - LOW) * 255
	skio.imsave(out_file, np.uint8(img))



###=========
def visualize_h5file_output_save(in_file: str, out_file: str):
	"""
	visualize the matlab output files. (generate ma) ********
	"""
	with h5py.File(in_file, 'r') as f:
		data_ct = np.asarray(f['gt_CT']).T
		data_ma = np.asarray(f['ma_CT'])[0].T
		metal_mask = np.asarray(f['metal_mask'])[0].T

	# LOW = np.min(data_ct)
	# HIGH = np.max(data_ct)

	data_ct = (data_ct - LOW) / (HIGH - LOW) * 255
	data_ma = adjust_ct_window(data_ma, minval=LOW, maxval=HIGH)
	data_ma = (data_ma - LOW) / (HIGH - LOW) * 255
	metal_mask = metal_mask * 255
	metal_mask[metal_mask>255] = 255

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

## -------------
if __name__ == "__main__":
	namelist = os.listdir(IN_FOLDER)
	if not os.path.exists(OUT_FOLDER):
		os.makedirs(OUT_FOLDER)
	for name in namelist:
		if not name.endswith(".h5"):
			continue
		in_file = os.path.join(IN_FOLDER, name)
		out_file = os.path.join(OUT_FOLDER, name.replace(".h5", ".png"))

		visualize_h5file_output_save(in_file=in_file, out_file=out_file)
		print("Draw the img:", name)

	print("End of the process.")