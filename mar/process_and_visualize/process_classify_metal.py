import os
import nibabel as nib
import numpy as np

'''
Classify metals and move them after processing ct with gt and no gt.

Move ct data to ma and noma folder depending on whether they have values over than the threshold.
Actually we can classify them after merging the two folders. 
'''
# ROOT_FOLDER = r'D:\ShenData\output_datasets\teeth300'
ROOT_FOLDER =  r"D:\ShenData\output_datasets_2204017\mergedata"

IN_GT_FOLDER = r'D:\ShenData\output_datasets_2204017\mergedata\with_label'
IN_NOGT_FOLDER = r'D:\ShenData\output_datasets_2204017\mergedata\no_label'

OUT_ROOT_FOLDER = r'D:\ShenData\output_datasets_2204017\merge_classified'

THRESHOLD = 2800

## process with GT
def classify_ma(in_folder: str, out_root_folder: str, threshod = 2500, with_gt = False):
	'''
	in_folder has such structure if with label gt:
		in_folder/
			29192839xxx-20200920.nii.gz
			29192839xxx-20200920_gt.nii.gz
			...
	or without gt:
		in_folder/
			29192839xxx-20200920.nii.gz
			...

	After process, the out_root_folder has structure:
		out_root_folder/
			withgt_withma/
				29192839xxx-20200920.nii.gz
				...
			withgt_withma_gt/
				29192839xxx-20200920_gt.nii.gz
				...
			withgt_noma/
				xxx-xxx.nii.gz
			withgt_noma_gt/
				xxx-xxx_gt.nii.gz

			nogt_withma/
				xxx-xxx.nii.gz
			nogt_noma/
				xxx-xxx.nii.gz

	'''
	namelist = [name.replace('.nii.gz', '') for name in os.listdir(in_folder) if not name.endswith('_gt.nii.gz')]
	namelist = list(set(namelist))

	if with_gt:
		out_ma_folder = os.path.join(out_root_folder, 'withlabel_withma')
		out_ma_gt_folder = os.path.join(out_root_folder, 'withlabel_withma_label')
		out_noma_folder = os.path.join(out_root_folder, 'withlabel_noma')
		out_noma_gt_folder = os.path.join(out_root_folder, 'withlabel_noma_label')
	else:
		out_ma_folder = os.path.join(out_root_folder, 'nolabel_withma')
		out_ma_gt_folder = ""
		out_noma_folder = os.path.join(out_root_folder, 'nolabel_noma')
		out_noma_gt_folder = ""
	
	if not os.path.exists(out_ma_folder):
		os.mkdir(out_ma_folder)
	if not os.path.exists(out_noma_folder):
		os.mkdir(out_noma_folder)
	if with_gt and not os.path.exists(out_ma_gt_folder):
		os.mkdir(out_ma_gt_folder)
	if with_gt and not os.path.exists(out_noma_gt_folder):
		os.mkdir(out_noma_gt_folder)

	for name in namelist:
		nii_filename = os.path.join(in_folder, name + '.nii.gz')
		gt_filename = os.path.join(in_folder, name+'_gt.nii.gz')
		img_array = nib.load(nii_filename).get_fdata()
		# get mask by threshod
		mask_value = np.sum(img_array > threshod)
		if mask_value > 0:
			# has metal
			out_filename = os.path.join(out_ma_folder, name + '.nii.gz')
			out_gt_filename = os.path.join(out_ma_gt_folder, name + '_gt.nii.gz')
			if with_gt:
				print("File %s has metal, move it to ma folder with gt." % name)
			else:
				print("File %s has metal, move it to ma folder (no gt)." % name)
			os.rename(nii_filename, out_filename)
			if with_gt:
				os.rename(gt_filename, out_gt_filename)
		else:
			# no mask
			out_filename = os.path.join(out_noma_folder, name + '.nii.gz')
			out_gt_filename = os.path.join(out_noma_gt_folder, name + '_gt.nii.gz')
			if with_gt:
				print("File %s has no metal, move it to noma folder with label." % name)
			else:
				print("File %s has no metal, move it to noma folder (no label)." % name)
			os.rename(nii_filename, out_filename)
			if with_gt:
				os.rename(gt_filename, out_gt_filename)

	print("Finish classification.")

if __name__ == '__main__':
	print("Classify files with label...")
	classify_ma(in_folder=IN_GT_FOLDER, out_root_folder=OUT_ROOT_FOLDER, threshod=THRESHOLD, with_gt=True)

	print("Classifying files with no label...")
	classify_ma(in_folder=IN_NOGT_FOLDER, out_root_folder=OUT_ROOT_FOLDER, threshod=THRESHOLD, with_gt=False)

	print("End of the process.")


