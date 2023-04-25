import os
import numpy as np
import nibabel as nib
import argparse
from PIL import Image
from tqdm import tqdm
#### --------------- 说明
'''
将文件夹里的所有nii文件全部进行分解，成多个小的文件，然后分成多个文件夹。类似视频一样。

# 22-1-21：现在讲train分成两个，一个train一个train2，用来将其中一个只提取内部部分（去掉两边区域），另一个完整提取
'''
mode = 'both'

data_input_folder = r"guozhen-data-fortest"
data_output_folder = r"output_test_folder\videos"


mask_input_folder = r"guozhen-anno-fortest"
mask_output_folder = r"output_test_folder\masks"










#### ---- 处理
def _image3D_normalize(image_data: np.ndarray):
    maxval = np.max(image_data)
    minval = np.min(image_data)
    normal_data = (image_data - minval) / (maxval - minval)

    # 导出了 [0,1] * 255 范围的数据
    return normal_data*255

def read_nii_to_3darray(nii_path: str, with_shape="WHC", mode="data" ):
    '''
    Read nii file to 3d numpy array with shape (x, y, z), default return shape is (x, y, z)
    nibabel read files with default shape: (x,y,z)
    '''
    nii_img = nib.load(nii_path)
    img_array = nii_img.get_fdata()

    ## with preproces
    if mode=='data':
        img_array = _image3D_normalize(img_array)
    elif mode=='mask':
        pass
    else:
        raise ValueError("mode "+mode+" not defined.")

    # adjust return shape
    if with_shape == "WHC":
        return img_array
    elif with_shape == "CHW":
        return img_array.transpose([2,1,0])
    else:
        raise ValueError("type " + str(with_shape) + "is wrong.")

def generate_random_index(length, omit_tail = False):
    """
    Generate random split index series, the last one is the end index. If `omit_tail` = True, then the head and tail part will be removed. 
    """
    if omit_tail == True:
        # 去掉头尾部分的空白序列
        begin = np.random.randint(100,200, 1)[0]
        # 最大是427层
        length = np.random.randint(400,450, 1)[0]
    else:
        begin = 0
    
    index_list = [begin ]
    while begin < length:
        end = begin + 20 + np.random.randint(0, 31, 1)[0]
        # 使最后一列的数量不少于20个
        if length - end < 20:
            end = length
        # 划分index序列
        if end >= length:
            index_list.append(length)
            break
        else:
            index_list.append(end)

        begin = end
    return index_list

def split_3darray(data: np.ndarray, split_indices: list = None):
    '''
    Split 3-d array with shape (C, H, W) to several slices.
    '''
    C, H, W = data.shape
    assert C >= 20, "Channel length is not enough."
    if split_indices:
        index_list = split_indices
    else:
        index_list = generate_random_index(C)

    slice_list = []
    K = len(index_list)

    for i in range(K-1):
        begin , end = index_list[i], index_list[i+1]
        slice_list.append(data[begin:end, :, :])
    
    return slice_list



def _old_split_3darray(data: np.ndarray, mode='random'):
    '''
    Split 3-d array with shape (C, H, W) to several slices.
    '''
    # NOTE: 这里有错误，image和mask随机分割的数量，需要调整进行切分
    C, H, W = data.shape
    assert C >= 20, "Channel number is not enough."
    ## 先按照随机方式分组，从20个开始筛选，最多到50个。
    slice_list = []
    begin = 0
    end = begin + 20 + np.random.randint(0, 31, 1)[0]
    slice_list = [ data[begin:end, :,:] ]
    begin = end
    while begin < C :
        end = begin + 20 + np.random.randint(0, 31, 1)[0]
        if end >= C:
            slice_list.append(data[begin: C, :, :])
        else:
            slice_list.append(data[begin:end, :, :])
        begin = end
    ## 比较最后两个效果结果
    last_len = slice_list[-1].shape[0]
    if last_len < 10:
        ## 想到的方法就是直接将最后一个和倒数第二个合并了
        last_slice = slice_list.pop(-1)
        slice_list[-1] = np.concatenate([slice_list[-1], last_slice] , axis=0)

    return slice_list

#### file io
def _save_2darray_to_image(img_array: np.ndarray, filename: str, filetype:str = 'data'):
    '''
    Save image array to file. img_array with shape: (H, W) or (C, H, W)
    '''
    if len(img_array.shape) == 3:
        img_array = img_array.transpose([1, 2, 0])
    img = Image.fromarray(np.uint8(img_array))
    img.save(filename)

def nii_to_multisegs(input_folder:str, output_folder: str, mode='data', split_indices: list = None):
    """
    Export nii files to multiple segments.
    """
    namelist = os.listdir(input_folder)
    for name in namelist:
        print("Processing file: ", name)
        filename = os.path.join(input_folder, name)
        data = read_nii_to_3darray(filename, with_shape='CHW', mode=mode)
        length = data.shape[0]
        if split_indices:
            data_list = split_3darray(data, split_indices)
        else:
            data_list = split_3darray(data)

        if name.endswith(".nii"):
            name = name.replace(".nii", "")
        elif name.endswith(".nii.gz"):
            name = name.replace(".nii.gz", "")
        else :
            continue

        # save subfiles
        with tqdm(total=length, desc="Processing:") as pbar:
            num = 0
            for i, subdata in enumerate(data_list):
                # TODO:这里的i不对应slice，经过剪裁后，还需要再修改
                subfolder = os.path.join(output_folder, name+"sub"+str(i) )
                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)
                C = subdata.shape[0]
                for each_slice in subdata:
                    outfile = os.path.join(subfolder, name+str(num)+".png")
                    _save_2darray_to_image(each_slice, outfile)

                    ## update
                    num += 1
                    pbar.update(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--data_input_folder", type=str, default=None)
    parser.add_argument("--data_output_folder", type=str, default=None)
    parser.add_argument("--mask_input_folder", type=str, default=None)
    parser.add_argument("--mask_output_folder", type=str, default=None)
    parser.add_argument("--omit_tail", action="store_true")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.data_input_folder:
        data_input_folder = args.data_input_folder
    if args.data_output_folder:
        data_output_folder = args.data_output_folder
    if args.mask_input_folder:
        mask_input_folder = args.mask_input_folder
    if args.mask_output_folder:
        mask_output_folder = args.mask_output_folder
    if args.mode:
        mode = args.mode
        
    print("mode: ", mode)
    print("data_input_folder: ", data_input_folder)
    print("data_output_folder: ", data_output_folder)
    print("mask_input_folder: ", mask_input_folder)
    print("mask_output_folder: ", mask_output_folder)

    length = 609
    
    # 掐头去尾
    if args.omit_tail:
        split_indices = generate_random_index(length, omit_tail=True)
    else:
        split_indices = generate_random_index(length)
    if mode in ('data', 'both'):
        nii_to_multisegs(data_input_folder, data_output_folder, mode='data', split_indices=split_indices)
    if mode in ('mask', 'both'):
        nii_to_multisegs(mask_input_folder, mask_output_folder, mode='mask', split_indices=split_indices)





if __name__ == '__main__':
    main()

    print("End of the process.")









