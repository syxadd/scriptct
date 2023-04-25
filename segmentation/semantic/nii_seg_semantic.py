import numpy as np
import nibabel as nib
import os
from tqdm import tqdm


'''
Convert nii segmentation file to segmentation masks. 
Choose all values > 0 as the segmentation mask. 
There is only one mask in the file. 

Input folder should contains the nii files.
'''

INPUT_FOLDER = ""
OUTPUT_FOLDER = ""







####  ---- Run
def get_all_files(folder: str, with_path: bool = False):
	if with_path:
		filelist = [os.path.join(folder, name) for name in os.listdir(folder)]
	else:
		filelist = os.listdir(folder)
	return filelist

def single_nii_to_semantic(filename: str, out_file: str):
	nii_img = nib.load(filename)
	img_array = nii_img.get_fdata()

	out_array = (img_array > 0) * 1
	out_img = nib.Nifti1Image(out_array, nii_img.affine)

	nib.save(out_img, out_file)

def convert_all_files(folder: str, out_folder: str):
	namelist = get_all_files(folder, with_path=False)

	for name in namelist:
		filename = os.path.join(folder, name)
		out_file = os.path.join(out_folder, name)
		print("Converting file: ", name)
		single_nii_to_semantic(filename, out_file)

if __name__ == "__main__":
	convert_all_files(INPUT_FOLDER, OUTPUT_FOLDER)
	print("End of the process.")