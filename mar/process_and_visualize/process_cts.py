import os
from tqdm import tqdm
import shutil

'''
Process all CTs in the folder. Detect the files with label and not with label. 
Then copy them to label folder or nolabel folder.

The folder contains the following structure:
```
folder/
	20xx1090/		(contains gt files)
		20200980.nii.gz
		20200980_gt.nii.gz
		20200980_predictionxx.gz
		...
	923839282/		(does not contain gt files)
		20200330.nii.gz
		20200330_prediction.nii.gz
	...


```

This file export all ct slices with masks to output folder.
'''

# IN_FOLDER = r'D:\ShenData\teeth_dataset300'
IN_FOLDER = r'D:\ShenData\teeth_dataset300'

OUT_FOLDER = r"D:\ShenData\output_datasets_2204017\teeth_dataset300"

# count = 0
# gt_count = 0
# nogt_count = 0

# allnames = []
# allfolders = []

_path_start = len(IN_FOLDER)
def get_all_ctfiles(in_folder: str):
	all_names = []
	count = 0
	gt_count = 0

	for root, folders, names in os.walk(in_folder):
		current_folder = root[_path_start+1:]
		# print(current_folder)
		
		namelist = []
		for name in names:
			if name[0] == '.':
				continue
			# name, _ = os.path.splitext(name)
			if name.endswith("gt.nii.gz"):
				gt_count += 1

			name = name[:8]
			name = os.path.join(current_folder, name)
			namelist.append(name)
		namelist = list(set(namelist))
		count += len(namelist)
		all_names.extend(namelist)

	## output infos 
	print("Total files have number: ", len(all_names))
	print("Total ct files: ", count)
	print("Total gt counts: ", gt_count)

	return all_names

def filter_gt_files(folder: str, namelist):
	'''
	namelist contains all ct name numbers including prefix folder
	'''
	gt_list = []
	gt_count = 0
	nongt_count = 0
	nongt_list = []
	for name in namelist:
		if os.path.exists(os.path.join(folder, name) + '_gt.nii.gz'):
			gt_list.append(name)
			gt_count +=1
		else:
			nongt_list.append(name)
			nongt_count += 1
	
	print("Total gt files : ", gt_count)
	print("Total nongt files: ", nongt_count)
	return {
		"gt_names": gt_list,
		"nongt_names": nongt_list
	}

# def filter_ma_files(folder: str, namedict: dict):
# 	'''
# 	namedict is obtained by the above process.
# 	'''
# 	gt_list = namedict['gt_names']
# 	nongt_list = namedict['nongt_names']
# 	# process with gt
# 	for name in gt_list:
# 		pass


	


# _-------------------
if __name__ == '__main__':

	allnames = get_all_ctfiles(in_folder=IN_FOLDER)
	if not os.path.exists(OUT_FOLDER):
		os.makedirs(OUT_FOLDER)

	## process allnames 
	namedict = filter_gt_files(IN_FOLDER, allnames)

	## copy files with label
	out_gt_folder = os.path.join(OUT_FOLDER, 'with_label')
	if not os.path.exists(out_gt_folder):
		os.mkdir(out_gt_folder)
	gt_list = namedict['gt_names']
	gt_count = 0
	for name in gt_list:
		# name has format "20398434\\20200901", outname has format "20398434-20200901"
		outname = name.replace('\\', '-')  
		name = os.path.join(IN_FOLDER, name)
		outname = os.path.join(out_gt_folder, outname)
		try:
			shutil.copy(name+'.nii.gz', outname+'.nii.gz')
			shutil.copy(name+'_gt.nii.gz', outname+'_gt.nii.gz')
			gt_count += 1
		except FileNotFoundError:
			print("Warning: ", name, " not found.")
	
	print("Copied all gt files. Total ", gt_count)

	## copy files with no label
	out_nongt_folder = os.path.join(OUT_FOLDER, 'no_label')
	if not os.path.exists(out_nongt_folder):
		os.mkdir(out_nongt_folder)
	nongt_list = namedict['nongt_names']
	nongt_count = 0
	for name in nongt_list:
		outname = name.replace('\\', '-')
		name = os.path.join(IN_FOLDER, name)
		outname = os.path.join(out_nongt_folder, outname)
		try:
			shutil.copy(name+'.nii.gz', outname+'.nii.gz')
			nongt_count += 1
		except FileNotFoundError:
			print("Warning: ", name, " not found.")
			
	
	print("Copied all non-gt files. Total ", nongt_count)
	

	print("End of the process. ")



# print("Total files have number: ", len(allnames))
# print("Total ct files: ", count)
# print("Total gt counts : ", gt_count)
# print("Total folders : ", len(allfolders))

