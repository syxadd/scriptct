import os
import numpy as np
from PIL import Image

'''
This file read annotated images and extract their class mask to png mask file.
Mainly process bmp files. 
'''
#%%
input_folder = r"data-all\已标注的图\sample"

output_folder = r"preparedata\tooth\output-multi-mask"


#%%
def read_image_tonumpy(filename: str):
    '''Read an image in numpy format'''
    img = Image.open(filename)
    # 转换成np.ndarray格式
    img = np.asarray(img)
    return img

def get_imagecolor_mask(img: np.ndarray):
    '''Extract color part of the image with a mask.
    Args:
        img: numpy array with shape (H, W, 3)
    Return:
        mask: numpy array with shape (H, W)
    '''
    ## 提取颜色部分mask
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]
    # 提取彩色部分, 灰色部分自然就是r=g=b了
    mask = (img_r == img_g) & (img_g == img_b) & (img_r == img_b) # bool type
    mask = ~mask
    # 去掉底部的红字(g,b==0)， 不过貌似600以下都是灰度了，所以直接去掉
    mask[600:, :] = False
    # 最终的0-1值的mask
    mask = 1 * mask
    return mask

def get_image_classes(img: np.ndarray):
    mask = get_imagecolor_mask(img)
    # mask dtype is int32
    subimg = img * ( np.array([mask, mask, mask]).transpose((1,2,0)) )
    # subimg with dtype int32

    ## 区分颜色，这里采用rgb加和方式来区分颜色
    ## NOTE: 这里的分类直接从1开始编号分类了，之后需要再调整成为每个牙的情况
    pixels = subimg.reshape(-1, 3)
    colors = np.unique(pixels, axis=0)
    # mask = np.zeros(img.shape)
    for i in range( len(colors)):
        if i == 0:
            continue
        color = colors[i]
        submask = (subimg[:,:,0] == color[0]) &  (subimg[:,:,1] == color[1]) & (subimg[:,:,2] == color[2])
        mask[submask] = i

    return mask

def convert_single_img2multiclass_mask(imgfile: str, outfile: str):
    img = read_image_tonumpy(imgfile)
    mask = get_image_classes(img)

    maskimg = Image.fromarray(np.uint8(mask))
    maskimg.save(outfile)

def convert_allimgs2classmask(in_folder: str, out_folder: str):
    '''
    Convert all images in in_folder to masks.
    '''
    assert input_folder != output_folder, "Output folder cannot be input folder"
    # filelist = [os.path.join(in_folder, name) for name in os.listdir(in_folder)]
    filelist = os.listdir(in_folder)

    for filename in filelist:
        name, ext = os.path.splitext(filename)
        if ext == '.bmp':
            ext = '.png'
            name += ext
            out_file = os.path.join(out_folder, name)
            print("Output file is :", out_file)
            convert_single_img2multiclass_mask(filename, out_file)

    print("End of the process. ")

def main():
    print("Start to convert multiclass masks.")
    convert_allimgs2classmask(input_folder, output_folder)

if __name__=='__main__':
    img_name = r"Axial+0058.5-000.bmp"
    outfile = "result_sample.png"
    convert_single_img2multiclass_mask(img_name, outfile)


