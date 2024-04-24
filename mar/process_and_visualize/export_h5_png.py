import os
from typing import Optional
import numpy as np
import h5py
from tqdm import tqdm
import cv2

"""
Visualize simulation data and export the data to png files.

这个脚本目前用来展示导出的metal mask，导出分割mask暂时不可用；
"""

# IN_FOLDER = r"/public/bme/home/v-xujun/syx-working/output_datasets_2204017/export_video_slices221101/withlabel_noma"
# OUT_FOLDER = r"/public/bme/home/v-xujun/project_mar2new/paired_data_segmentation"
IN_FOLDER = r"/public/bme/home/v-xujun/syx-working/output_datasets_2204017/export_video_slices221101/withlabel_noma"
OUT_FOLDER = r"/public/bme/home/v-xujun/project_mar2new/paired_data_segmentation"

mode = "origin"
# mode : "simulated", "origin"



#### ------ functions
def get_all_files(folder: str):
    # name does not contain the root folder
    filelist = []
    for root, _, names in os.walk(folder):
        folder_prefix = root[len(folder)+1:]
        for name in names:
            if name.endswith(".h5"):
                if len(folder_prefix) > 0:
                    filename = folder_prefix + '/' + name
                else:
                    filename = name
                filelist.append(filename)
    return filelist

def get_minmax():
    return 0.0, 0.73

def get_window():
    return -1000, 2800

def normalize(data, minmax):
    minval, maxval = minmax
    data = np.clip(data, minval, maxval)
    data = (data - minval) / (maxval - minval)
    data = data * 2.0 - 1.0
    return data

# def denormalize(data, minmax):
#     minval, maxval = minmax
#     data = data * 0.5 + 0.5
#     data = data * (maxval - minval) + minval
#     return data

def normalize_only(data, minmax):
    minval, maxval = minmax
    # data = np.clip(data, minval, maxval)
    data = (data - minval) / (maxval - minval)
    data = data * 2.0 - 1.0
    return data

def read_simulated_datah5(filename: str, withnorm=False, window=None):
    '''
    Use for test process. Get ndarray back. Manually load to tensor please. 
    This function does not normalize the read data.
    '''
    with h5py.File(filename, mode='r') as f:
        data_gt = np.array(f['gt_CT']) # f['gt_CT'] with shape (H, W)
        data_ma = np.array(f['ma_CT']) # f['ma_CT'] with shape (1, H, W)
        data_metalmask = np.array(f['metal_mask']) # f['metal_mask'] with shape (1, H, W)
        data_ma = data_ma[0]
        data_metalmask = data_metalmask[0]
        
    if withnorm:
        if window is not None:
            miuwater = 0.192
            minval, maxval = window
            minval = minval / 1000 * miuwater + miuwater
            maxval = maxval / 1000 * miuwater + miuwater
        else:
            minval, maxval = get_minmax()


        data_gt = normalize(data_gt, (minval, maxval))
        data_ma = normalize(data_ma, (minval, maxval))

    # return all data with single image shape (H, W)
    data = {
        "gt_image": data_gt,
        "ma_image": data_ma,
        "metal_mask": data_metalmask,
        "filename": filename
    }
    return data


def read_datah5_exslice(filename: str, withnorm: bool, istrunc: bool, window_size: Optional[tuple] = None, mask_threshold: Optional[float] = None):
    '''
    Read the exported slices from nii files. So it should have `image` and `mask`.
    '''
    with h5py.File(filename, mode='r') as f:
        data_ma = np.array(f['image']) # read with shape (H, W)
        data_mask = np.asarray(f['mask'])

        # if mask_threshold is None:
        #     data_metalmask = np.array(f['metal_mask']) # read with shape (H, W)
        # else:
        #     data_metalmask = data_ma > mask_threshold

    miuwater = 0.192
    data_ma = data_ma / 1000 * miuwater + miuwater

    if withnorm :
        if window_size is not None:
            minval, maxval = window_size
            minval = minval / 1000 * miuwater + miuwater
            maxval = maxval / 1000 * miuwater + miuwater
        else:
            minval, maxval = get_minmax()

        # data_ma = (data_ma - minval) / (maxval - minval)
        if istrunc:
            data_ma = normalize(data_ma, (minval, maxval))
        else:
            data_ma = normalize_only(data_ma, (minval, maxval))


    # return all data with single image shape (H, W)
    data = {
        "filename": filename,
        "ma_image": data_ma,
        "mask": data_mask,
        # "metal_mask": data_metalmask,

    }
    return data

def is_chinese(s: str):
    # From : https://blog.csdn.net/qdPython/article/details/110231244
    for ch in s:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    
    return False




##### ------- process
def get_simulated_image(h5_file: str):
    data = read_simulated_datah5(h5_file, withnorm=True)
    # data in range [-1, 1]
    gt_image = data['gt_image'] * 0.5 + 0.5
    ma_image = data['ma_image'] * 0.5 + 0.5
    ma_mask = data['metal_mask'] > 0

    # [0,1] to [0, 255]
    gt_image = (gt_image * 255.0).round()
    ma_image = (ma_image * 255.0).round()
    ma_mask = ma_mask * 255

    # 
    outputs = {
        "gt_image": np.uint8(gt_image),
        "ma_image": np.uint8(ma_image),
        "ma_mask": np.uint8(ma_mask),
    }
    return outputs

def get_origin_slice_image(h5_file: str):
    data = read_datah5_exslice(h5_file, withnorm=True, istrunc=True)
    # rescale
    ma_image = data['ma_image'] * 0.5 + 0.5
    mask = data['mask'] > 0

    # [0, 1] to [0, 255]
    ma_image = (ma_image * 255.0).round()
    mask = mask * 255

    outputs = {
        "ma_image": np.uint8(ma_image),
        "mask": np.uint8(mask),
    }

    return outputs 




def main_simulated():
    in_folder = IN_FOLDER
    out_folder = OUT_FOLDER
    
    # out_gt_dir = os.path.join(out_folder, "images_gt")
    # out_ma_dir = os.path.join(out_folder, "images_ma")
    # out_mask_dir = os.path.join(out_folder, "masks")
    # out_ma_mask_dir = os.path.join(out_folder, "metal_masks")
    # if not os.path.exists(out_gt_dir):
    #     os.makedirs(out_gt_dir)
    # if not os.path.exists(out_ma_dir):
    #     os.makedirs(out_ma_dir)
    # if not os.path.exists(out_mask_dir):
    #     os.makedirs(out_mask_dir)
    # if not os.path.exists(out_mask_dir):
    #     os.makedirs(out_mask_dir)

    out_folder_gt = os.path.join(out_folder, "GT")
    out_folder_ma = os.path.join(out_folder, "MA")
    out_folder_mamask = os.path.join(out_folder, "ma_mask")

    if not os.path.exists(out_folder_gt):
        os.makedirs(out_folder_gt)
    if not os.path.exists(out_folder_ma):
        os.makedirs(out_folder_ma)
    if not os.path.exists(out_folder_mamask):
        os.makedirs(out_folder_mamask)

    namelist = get_all_files(in_folder)

    for name in tqdm(namelist):
        h5_file = os.path.join(in_folder, name)

        outputs = get_simulated_image(h5_file)


        gt_image = outputs['gt_image']
        ma_image = outputs['ma_image']
        ma_mask = outputs['ma_mask']

        name = name.replace('/', '-')
        name = name.replace('\\', '-')
        out_file_gt = os.path.join(out_folder_gt, name+".png")
        out_file_ma = os.path.join(out_folder_ma, name+".png")
        out_file_mamask = os.path.join(out_folder_mamask, name+".png")
        cv2.imwrite(out_file_gt, gt_image)
        cv2.imwrite(out_file_ma, ma_image)
        cv2.imwrite(out_file_mamask, ma_mask)

    print("End of the process.")



def main_origin():
    in_folder = IN_FOLDER
    out_folder = OUT_FOLDER

    # out_folder_gt = os.path.join(out_folder, "GT")
    out_folder_ma = os.path.join(out_folder, "MA")
    out_folder_mask = os.path.join(out_folder, "mask")

    # if not os.path.exists(out_folder_gt):
    #     os.makedirs(out_folder_gt)
    if not os.path.exists(out_folder_ma):
        os.makedirs(out_folder_ma)
    if not os.path.exists(out_folder_mask):
        os.makedirs(out_folder_mask)

    namelist = get_all_files(in_folder)

    for name in tqdm(namelist):
        h5_file = os.path.join(in_folder, name)

        outputs = get_origin_slice_image(h5_file)

        ma_image = outputs['ma_image']
        mask = outputs['mask']

        name = name.replace('/', '-')
        name = name.replace('\\', '-')
        out_file_ma = os.path.join(out_folder_ma, name+".png")
        out_file_mask = os.path.join(out_folder_mask, name+".png")
        cv2.imwrite(out_file_ma, ma_image)
        cv2.imwrite(out_file_mask, mask)


    print("End of the process.")



if __name__ == "__main__":
    if mode == "simulated":
        main_simulated()
    elif mode == "origin":
        main_origin()