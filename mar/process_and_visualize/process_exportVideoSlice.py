import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import h5py
import skimage.io as skio

'''
Find the tooth slice with segmentation and randomly select several metals in each whole nii file. 

Then export the continuous slices with tooth and the metal mask. If the origin teeth contain metal, then we ignore these slices and make a new random selection. 


After the process, we need to scan all folders to find the folder that contains only single slice (or no more than 3 ? ) and remove the folder.

Each CT scan nii file will export ma slices and noma slices to 'ma' folder and 'noma' folder.


Note: 

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
IN_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel_withma_valid"
GT_FOLDER = r"D:\ShenData\output_datasets_2204017\merge_classified\withlabel_withma_label_valid"




# OUT_ROOT_FOLDER = r"D:\ShenData\output_datasets_2204017\export_slices\data_withlabel_noma"
# OUT_ROOT_FOLDER = r"D:\ShenData\output_datasets_2204017\export_video_slices221101\withlabel_train"

OUT_ROOT_FOLDER = r"D:\ShenData\output_datasets_2204017\export_video_slices221101\valid"


THRESHOLD = 2800


#### no GT path
# IN_FOLDER = r"D:\ShenData\output_datasets\mergedata\nolabel_withma"

# OUT_ROOT_FOLDER = r"D:\ShenData\output_datasets\export_slices\data_nolabel"
# GT_FOLDER = ""

#### --------
seed = 100
np.random.seed(seed)

WITH_GT = False
if GT_FOLDER and len(GT_FOLDER) > 0:
	WITH_GT = True



#### ----

def export_videoslices_withlabel_withma(img_file: str, gt_file: str, out_folder: str, threshold=2800, visualize=False):
	'''
	Export the video-like slices and drop the real ma slices. When there exists real ma, it will split the nii file into several sub consecutive slices like video.
	`out_folder` is the root of the saved data; `threshold` is used to find real metals.
	'''

	img_array = nib.load(img_file).get_fdata()
	gt_array = nib.load(gt_file).get_fdata()
	x,y,z = img_array.shape
	name = os.path.split(img_file)[-1]
	name = name.replace(".nii.gz", "")



	# if not os.path.exists(out_ma_folder):
	# 	os.mkdir(out_ma_folder)
	# if not os.path.exists(out_noma_folder):
	# 	os.mkdir(out_noma_folder)
	
	# obj_folder = os.path.join(out_ma_folder, name)
	# if not os.path.exists(obj_folder):
	# 	os.mkdir(obj_folder)
	# obj_folder = os.path.join(out_noma_folder, name)
	# if not os.path.exists(obj_folder):
	# 	os.mkdir(obj_folder)

	## random select several teeth as metals
	uniques = np.sort(np.unique(gt_array))[1:]
	uniques = uniques.astype(np.int)
	metal_number = np.random.randint(low=3, high=10)
	metal_ids = np.random.choice(uniques, size=metal_number, replace=False)
	metal_ids = metal_ids.astype(gt_array.dtype)
	metal_volume = np.isin(gt_array, metal_ids)  # metal_mask
	

	## 判断整个是否有metal
	metal = img_array > threshold
	ma_slices = metal.sum(axis=(0, 1))
	# if (ma_slices > 10).sum() > 0:
	# 	is_ma = True
	# else:
	# 	is_ma = False
	

	# create output folder
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	out_ma_folder = os.path.join(out_folder, "ma")
	out_noma_folder = os.path.join(out_folder, "noma")
	if not os.path.exists(out_ma_folder):
		os.mkdir(out_ma_folder)
	if not os.path.exists(out_noma_folder):
		os.mkdir(out_noma_folder)
	## create output file folder to save slices
	obj_name = name+"(---)"
	# if not os.path.exists(obj_folder):
	# 	os.makedirs(obj_folder)

	# this value decides whether to create new folder for current videos 
	whether_newfolder = False
	folder_id = 0
	## construct new videos 
	for s in range(z):
		is_gt = False
		is_ma = False
		img = img_array[:,:,s].T
		mask = gt_array[:,:,s].T
		metal_mask = metal_volume[:,:,s].T
		# find whether has gt label (whether to have tooth object)
		if np.sum(mask > 0) > 0:
			# save image slice and mask slice
			is_gt = True
		else:
			continue
		# find whether has metal, here adjust if it is the metal with value area over 10
		if ma_slices[s] > 10:
			is_ma = True
			whether_newfolder = True
			metal_mask = metal[:,:,s].T # real metal
			# continue
		else:
			is_ma = False
		
		## after the finding process, decide whether to build new folder
		if is_ma == False and whether_newfolder:
			# NOTE: to ensure the length enough, we assume the slices are no more than 999
			obj_name = obj_name[:-5]+f"({folder_id:0>3d})"
			folder_id += 1
			whether_newfolder = False


		
		## then export the slice to object folder
		if is_ma:
			obj_folder = os.path.join(out_ma_folder, obj_name)
		else:
			obj_folder = os.path.join(out_noma_folder, obj_name)
		if not os.path.exists(obj_folder):
			os.makedirs(obj_folder)
		# save data in HDF5 file (h5py)
		out_file = os.path.join(obj_folder, str(s)+'.h5')
		with h5py.File(out_file, mode='w') as f:
			f.create_dataset(name="image", data=img, dtype=int, compression='gzip', compression_opts=6)
			f.create_dataset(name="mask", data=mask, dtype=int, compression='gzip', compression_opts=6)
			f.create_dataset(name="metal_mask", data=metal_mask, dtype=int, compression='gzip', compression_opts=6)

		if visualize:
			out_file = os.path.join(obj_folder, str(s)+'_MAmask.png')
			skio.imsave(out_file, metal_mask*255)
		# end



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
			if name.endswith("gt.nii.gz"):
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
		
		for name in tqdm(nii_files):
			filename = os.path.join(IN_FOLDER, name)
			subname = name.replace(".nii.gz", "")
			# out_folder = os.path.join(OUT_ROOT_FOLDER, subname)

			gt_file = os.path.join(GT_FOLDER, subname+'_gt.nii.gz')

			export_videoslices_withlabel_withma(filename, gt_file=gt_file, out_folder=OUT_ROOT_FOLDER, threshold=THRESHOLD, visualize=True)

	else:
		# TODO: process files without GT
		raise NotImplementedError
		print("Processing data without label...")
		nii_files = get_all_files(IN_FOLDER)
		# nii_files = os.listdir(IN_FOLDER)
		if not os.path.exists(OUT_ROOT_FOLDER):
			os.makedirs(OUT_ROOT_FOLDER)
		for name in tqdm(nii_files):
			start_idx, end_idx = 77, 117

			filename = os.path.join(IN_FOLDER, name)
			subname, _ = os.path.splitext(name)
			# out_folder = os.path.join(OUT_ROOT_FOLDER, subname)

			gt_file = os.path.join(GT_FOLDER, name)

			export_slices_nolabel(filename, out_folder=OUT_ROOT_FOLDER, index_start=start_idx, index_end=end_idx, threshold=THRESHOLD)
