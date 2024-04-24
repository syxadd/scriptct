import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import h5py
import skimage.io as skio

'''
After classifying labeled and non-labeled images, we export image slices to folders. 
Each Image's slices are selected according to the annotation file if it has gt or slices are selected by a given index range. )

Each folder contains the slices of one image. 
(Well actually, we only classify the slice images using threshold, so it does not matter how each file is classified, just inputing all of them could also take effect. )

Assume the input folder has the following structure:
merged_folder_root/
	nolabel_withma/
		xxx-xxx.nii.gz
		...
	withlabel_withma/
		xxx-xxx.nii.gz
		...
	withlabel_withma_gt/
		xxx-xxx.nii.gz
		...
'''

#### with GT path
# valid
# IN_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_withma_valid"
# GT_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_withma_valid_label"
# OUT_ROOT_FOLDER = r"C:\Users\csjunxu\Documents\syx-working\project-marData\export_slices\data_withlabel_valid"

# test
IN_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_noma_test"
GT_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_noma_test_label"
# IN_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_withma_test"
# GT_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_withma_test_label"

OUT_ROOT_FOLDER = r"C:\Users\csjunxu\Documents\syx-working\project-marData\export_slices\data_withlabel_test"

# train
# IN_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_noma"
# GT_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_noma_label"
# IN_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_withma"
# GT_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel\withlabel_withma_label"

# OUT_ROOT_FOLDER = r"C:\Users\csjunxu\Documents\syx-working\project-marData\export_slices\data_withlabel_train"


# Select metal mask with threshold
THRESHOLD = 2800



#### no GT path  -- for data without GT -- such as clinical data
# IN_FOLDER = r"D:\syx-working\MAR2-结果整理与汇报\221009预测新的三个样本\RealData"
# GT_FOLDER = ""

# OUT_ROOT_FOLDER = r"D:\syx-working\MAR2-结果整理与汇报\221009预测新的三个样本\重构用H5文件"

# index_start, index_end = 171, 285  # data1
# index_start, index_end = 409, 569  # data2
# index_start, index_end = 275, 452  # data3

# index_start, index_end = 0, 1300  # all
# THRESHOLD = 2800

# output metal mask
VISUALIZE = True


#### --------
seed = 100
np.random.seed(seed)

WITH_GT = False
if GT_FOLDER and len(GT_FOLDER) > 0:
	WITH_GT = True


def export_slices_withlabel(img_file: str, gt_file: str, out_folder: str, threshold=2500, visualize=False):
	'''
	Export the slices. 
	out_folder is the root of the saved data.
	The function will save all slices of one data in a specific sub folder with index slice.
	'''
	# threshold=2500

	img_array = nib.load(img_file).get_fdata()
	gt_array = nib.load(gt_file).get_fdata()
	x,y,z = img_array.shape
	name = os.path.split(img_file)[-1]
	name = name.replace(".nii.gz", "")
	name = name.replace(".nii", "")
	# create folder
	out_ma_folder = os.path.join(out_folder, "ma")
	out_noma_folder = os.path.join(out_folder, "noma")
	if not os.path.exists(out_ma_folder):
		os.mkdir(out_ma_folder)
	if not os.path.exists(out_noma_folder):
		os.mkdir(out_noma_folder)
	
	obj_folder = os.path.join(out_ma_folder, name)
	if not os.path.exists(obj_folder):
		os.mkdir(obj_folder)
	obj_folder = os.path.join(out_noma_folder, name)
	if not os.path.exists(obj_folder):
		os.mkdir(obj_folder)

	## random select several teeth as metals
	# number = np.random.randint(low=0, high=5)
	# uniques = np.unique(gt_array)
	# uniques = uniques[uniques > 0]
	
	# metal_mask = gt_array 

	for s in tqdm(range(z)):
		is_gt = False
		is_ma = False
		img = img_array[:,:,s].T
		mask = gt_array[:,:,s].T
		# find whether has gt label
		if np.sum(mask > 0) > 0:
			# save image slice and mask slice
			is_gt = True
		else:
			continue
		# find whether has metal, here adjust if it is the metal with value area over 10
		if np.sum(img > threshold) > 10:
			is_ma = True
			metal_mask = (img > threshold)*1
		else:
			# contains no metal; so we generate metal
			uniques = np.unique(mask)[1:] # remove 0.0 , and get labels
			N = len(uniques)
			N = min(N, 12)
			# number = np.random.randint(low=1, high=5) #random select ma numbers in [1 ,5)
			number = 0
			while number <= 0:
				# p = 0.5
				number = np.random.binomial(N, 0.5)
			# if number > len(uniques):
			# 	# number = np.random.randint(low=1, high=len(uniques)+1)
			sample = np.random.choice(uniques, size=number, replace=False)
			metal_mask = np.isin(mask, sample) * 1

		if is_gt:
			if is_ma:
				obj_folder = os.path.join(out_ma_folder, name)
			else:
				obj_folder = os.path.join(out_noma_folder, name)
			# save data with numpy
			# data = {
			# 	"image": img,
			# 	"mask": mask,
			# }
			# out_file = os.path.join(obj_folder, str(s)+'.npz' )
			# np.savez_compressed(out_file, **data)

			# save data with h5py
			out_file = os.path.join(obj_folder, str(s)+'.h5')
			with h5py.File(out_file, mode='w') as f:
				f.create_dataset(name="image", data=img, dtype=int, compression='gzip', compression_opts=6)
				f.create_dataset(name="mask", data=mask, dtype=int, compression='gzip', compression_opts=6)
				f.create_dataset(name="metal_mask", data=metal_mask, dtype=int, compression='gzip', compression_opts=6)

		if visualize:
			out_file = os.path.join(obj_folder, str(s)+'_MAmask.png')
			skio.imsave(out_file, np.uint8(metal_mask*255), check_contrast=False)
			# end



def export_slices_nolabel(img_file: str, out_folder: str, index_start=0, index_end=999, threshold=2500, visualize=False):
	'''
	Export the slices. This function process nii files with no gt.
	out_folder is the root of the saved data.
	The function will save split slices [index_start, index_end) and save the middle slices of one data in a specific sub folder with index slice.
	'''
	img_array = nib.load(img_file).get_fdata()

	x,y,z = img_array.shape
	name = os.path.split(img_file)[-1]
	name = name.replace(".nii.gz", "")
	name = name.replace(".nii", "")
	if index_end > z:
		index_end = z
	# create folder
	out_ma_folder = os.path.join(out_folder, "ma")
	out_noma_folder = os.path.join(out_folder, "noma")
	if not os.path.exists(out_ma_folder):
		os.mkdir(out_ma_folder)
	if not os.path.exists(out_noma_folder):
		os.mkdir(out_noma_folder)
	
	obj_folder = os.path.join(out_ma_folder, name)
	if not os.path.exists(obj_folder):
		os.mkdir(obj_folder)
	obj_folder = os.path.join(out_noma_folder, name)
	if not os.path.exists(obj_folder):
		os.mkdir(obj_folder)
	# out_folder = os.path.join(out_folder, name)
	# if not os.path.exists(out_folder):
	# 	os.mkdir(out_folder)

	for s in tqdm(range(index_start, index_end)):
		is_gt = False
		is_ma = False
		img = img_array[:,:,s].T

		if np.sum(img > threshold) > 10:
			is_ma = True


		metal_mask = (img > threshold) * 1
		if is_ma:
			obj_folder = os.path.join(out_ma_folder, name)
		else:
			obj_folder = os.path.join(out_noma_folder, name)

		# data = {
		# 	"image": img,
		# 	"mask": np.array([]),
		# }
		# out_file = os.path.join(obj_folder, str(s)+'.npz' )
		# np.savez_compressed(out_file, **data)
		# save data with h5py
		out_file = os.path.join(obj_folder, str(s)+'.h5')
		with h5py.File(out_file, mode='w') as f:
			f.create_dataset(name="image", data=img, dtype=int, compression='gzip', compression_opts=6)
			f.create_dataset(name="metal_mask", data=metal_mask, dtype=int, compression='gzip', compression_opts=6)


		if visualize:
			out_file = os.path.join(obj_folder, str(s)+'_MAmask.png')
			skio.imsave(out_file, np.uint8(metal_mask*255), check_contrast=False)


def get_all_files(in_folder: str):
	"""
	Read all files in the folder, and return the filelist. 
	Filename only contains subfolder name. 
	"""
	_path_start = len(in_folder)
	all_names = []
	count = 0
	gt_count = 0

	for root, folders, names in os.walk(in_folder):
		# remove the pre root path, also works if in the root of folder
		current_folder = root[_path_start+1:] # contains a '/'
		# print(current_folder)
		
		namelist = []
		for name in names:
			if name[0] == '.':
				continue
			# name, _ = os.path.splitext(name)
			if name.endswith(("gt.nii.gz", "gt.nii")):
				gt_count += 1

			# name = name[:8]
			name = os.path.join(current_folder, name)
			namelist.append(name)
		# namelist = list(set(namelist))
		count += len(namelist)
		all_names.extend(namelist)

	## output infos 
	print("Total files have number: ", len(all_names))
	print("Total ct files: ", count)
	print("Total gt counts: ", gt_count)

	return all_names

if __name__ == '__main__':
	# get all files in the foler
	
	## process the files with GT
	if WITH_GT:
		print("Processing data with label...")
		nii_files = get_all_files(IN_FOLDER)
		# nii_files = os.listdir(IN_FOLDER)
		if not os.path.exists(OUT_ROOT_FOLDER):
			os.makedirs(OUT_ROOT_FOLDER)
		
		for name in nii_files:
			filename = os.path.join(IN_FOLDER, name)
			subname = name.replace(".nii.gz", "")
			# subname = name.replace(".nii", "")
			# out_folder = os.path.join(OUT_ROOT_FOLDER, subname)

			gt_file = os.path.join(GT_FOLDER, subname+'_gt.nii.gz')

			print(f"Processing file : {filename}")

			export_slices_withlabel(filename, gt_file=gt_file, out_folder=OUT_ROOT_FOLDER, threshold=THRESHOLD, visualize=VISUALIZE)

	else:
	## process files without GT
		print("Processing data without label...")
		nii_files = get_all_files(IN_FOLDER)
		# nii_files = os.listdir(IN_FOLDER)
		if not os.path.exists(OUT_ROOT_FOLDER):
			os.makedirs(OUT_ROOT_FOLDER)
		for name in nii_files:
			start_idx, end_idx = index_start, index_end

			filename = os.path.join(IN_FOLDER, name)
			subname, _ = os.path.splitext(name)
			# out_folder = os.path.join(OUT_ROOT_FOLDER, subname)
			# gt_file = os.path.join(GT_FOLDER, name)


			print(f"Processing file : {filename}")

			export_slices_nolabel(filename, out_folder=OUT_ROOT_FOLDER, index_start=start_idx, index_end=end_idx, threshold=THRESHOLD, visualize=VISUALIZE)
