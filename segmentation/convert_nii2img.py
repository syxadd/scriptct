import numpy as np
import os
import nibabel as nib
import imageio
from PIL import Image
import tqdm

## ---------------
'''
主要处理nii文件，转换成png，区分image和mask两种处理类型（对应原图像和标注数据segment）；

需要转换原数据就填image路径，转换标注就填mask路径。

注意：处理的的数据需要是nii和nii.gz。（如果导出图片太浅，需要单独调整一下窗宽window.）

在下边修改模式和文件位置.
mode : 
    data - ct数据，只处理img路径；
    mask - ct标注，只处理mask路径；
    both - 两个都处理，需要额外处理路径，第一个路径为image，第二个路径为mask
nii_img_path:
    CT影像的原路径，不需要就填空。如果 nii_path 是文件，则导出文件，如果是路径，则把路径下所有的nii文件全部导出。
output_img_folder：
    导出的图片的文件夹，不需要就填空。
nii_mask_path：
    标注数据的路径，如果只转换影像，不需要标注就填空。如果 nii_path 是文件，则导出文件，如果是路径，则把路径下所有的nii文件全部导出。
output_mask_folder：
    导出的标注的文件夹，不需要就填空。

WINDOW: 对于一些特定部位的可视化图像需要额外设置窗宽，以HU为单位。
'''

#### Options
mode = 'data'

nii_img_path = r"file.nii.gz"
output_img_folder = r"export-img"

nii_mask_path = ""
output_mask_folder = ""


# 默认(min, max) 
WINDOW = (None, None) 





#### ---- Run
def read_niifile(nii_filepath: str):
    # return 3D image
    img = nib.load(nii_filepath)  # 读取nii
    return img

def _image3D_normalize(image_data: np.ndarray, window=(None, None)):
    if window[0] is not None:
        minval = window[0]
    else:
        minval = np.min(image_data)
    if window[1] is not None:
        maxval = window[1]
    else:
        maxval = np.max(image_data)
    normal_data = (image_data - minval) / (maxval - minval)
    normal_data = np.clip(normal_data, 0, 1)

    # 导出了 [0,1] * 255 范围的数据
    return normal_data*255

    

#### ---- 转换
def _single_nii2image(nii_file: str, save_path: str, mode: str):
    '''
    Convert single nii 3D image to 2d slice images. 
    nii_file and save_path must be complete file path.
        mode: with 'data' mode and 'mask' mode.
    '''
    assert mode in ('data', 'mask'), "mode is wrong!"
    ## read data
    nii_img = nib.load(nii_file)
    img_fdata = nii_img.get_fdata()
    _, fname = os.path.split(nii_file)
    if fname.endswith('.nii'):
        fname = fname.replace('.nii', '')  # 去掉nii的后缀名
    elif fname.endswith('.nii.gz'):
        fname = fname.replace('.nii.gz', '') 
    img_f_path = os.path.join(save_path, fname)
    ## 保存目录新建
    if not os.path.exists(save_path):
        os.mkdir(save_path)  # 新建文件夹

    # prepocess of data
    if mode == 'data':
        # normalize data in [0, 255]
        img_fdata = _image3D_normalize(img_fdata, WINDOW)


    ## nibabel read with shape (x, y, z)
    (w, h, z) = nii_img.shape
    
    for i in tqdm.tqdm(range(z)):
        data_slice = img_fdata[:,:,i]
        #### 这里对提取的切片进行操作和保存
        # 这里提取的是x,y维度，为了获取h,w维度，需要进行转置
        data_slice = data_slice.T

        ## ** 这里处理mask，如果需要做语义分割，就把标注设为0，1
        if mode == 'mask':
            data_slice = data_slice
            # NOTE: 语义分割的处理方法
            # data_slice = data_slice > 0


        ## save data_slice
        filename = img_f_path+ '{}.png'.format(i)  # 每个文件尾部名字保存
        # 用imageio来保存切片（会将切片进行放缩）
        # imageio.imwrite(filename, data_slice)
        # 使用pil保存切片，首先已经经过了标准化

        ## NOTE: 语义分割的处理方法 -- 二选一
        if mode == 'mask':
            img_slice = Image.fromarray(np.uint8(data_slice)) 
            # NOTE: For Semantic use
            # img_slice = Image.fromarray(data_slice) 
        else:
            # data mode
            img_slice = Image.fromarray(np.uint8(data_slice))
        
        ## NOTE: 实例的处理方法 -- 二选一
        # img_slice = Image.fromarray(np.uint8(data_slice))


        img_slice.save(filename)
        # print("Image slice %d has been saved. " % i)
    print("Convert total %d images." % z)


def convert_nii2images(in_path: str, out_path: str, mode: str):
    '''
    Convert 3d nii files to images. mode is 'data' or 'mask'
    '''
    if os.path.isdir(in_path):
        # nii_path 为目录形式，导出所有的图
        print("Process all files in %s" % (in_path))
        filelist = [os.path.join(in_path, name) for name in os.listdir(in_path)]
        for filename in filelist:
            if not filename.endswith( ('.nii', '.nii.gz') ):
                continue
            print("Converting file: "+ filename)
            if mode == 'data':
                _single_nii2image(filename, out_path, mode='data')
            elif mode == 'mask':
                _single_nii2image(filename, out_path, mode='mask')
            else:
                raise NotImplementedError
    
    else:
        # nii_path 为文件形式，导出单个文件的图
        if mode == 'data':
            _single_nii2image(in_path, out_path, mode='data')
        elif mode == 'mask':
            _single_nii2image(in_path, out_path, mode='mask')
        else:
            raise NotImplementedError


if __name__ == '__main__':

    print("input file is : " + nii_img_path)
    print("Output folder is :" + output_img_folder)

    if mode == 'data':
        convert_nii2images(nii_img_path, output_img_folder, mode='data')
    elif mode == 'mask':
        convert_nii2images(nii_img_path, output_mask_folder, mode='mask')
    elif mode == 'both':
        # data
        convert_nii2images(nii_img_path, output_img_folder, mode='data')
        # mask
        convert_nii2images(nii_mask_path, output_mask_folder, mode='mask')

    print("-"*15)
    print("Process finished! mode: " + mode)
