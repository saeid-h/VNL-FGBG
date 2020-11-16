import os
import cv2
import torch
import numpy as np
# import tqdm
from lib.core.config import cfg
from lib.utils.net_tools import load_ckpt
from tools.parse_arg_test import TestOptions
from lib.core.config import merge_cfg_from_file
from data.load_dataset import CustomerDataLoader
from lib.models.image_transfer import resize_image
from lib.utils.evaluate_depth_error import evaluate_err
from lib.models.metric_depth_model import MetricDepthModel
from lib.utils.logging import setup_logging, SmoothedValue
import matplotlib.pyplot as plt
try:
    from imageio import imsave, imread
except:
    from scipy.misc import imsave, imread

logger = setup_logging(__name__)

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """    
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def depth_write(filename, depth):
    """ Write depth to file. """
    height,width = depth.shape[:2]
    f = open(filename,'wb')
    np.array(TAG_FLOAT).astype(np.float32).tofile(f)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    depth.astype(np.float32).tofile(f)
    f.close()

def sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # load model
    model = MetricDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    with open(test_args.file_list, 'r') as f:
            lines = f.readlines()
    img_names = [line.split()[0].replace(os.sep, '_') for line in lines]
    img_names.sort()
    # with open('lists/replica_test_files_with_gt.txt', 'r') as f:
    #     lines = f.readlines()
    # name_dict = {line.split()[0].split('_')[-1].split('.')[0]: line.split()[0].replace(os.path.sep, '_') for line in lines}

    model_name = test_args.load_ckpt.split('/')[-1].split('.')[0]
    outpath = os.path.join('results', model_name)
    os.system ('mkdir -p '+os.path.join(outpath, 'rgb'))
    os.system ('mkdir -p '+os.path.join(outpath, 'raw'))
    os.system ('mkdir -p '+os.path.join(outpath, 'cmap')) 
    os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_init'))
    os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_final'))
    os.system ('mkdir -p '+os.path.join(outpath, 'occ_mask_gt'))
    
    image_dir = os.path.join(outpath, 'raw')

    for i, data in enumerate(data_loader):
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['b_fake']).cpu().numpy()
        img_name = img_names[i]

        gt = data['B_raw'].cpu().numpy().squeeze()

        depth_write(os.path.join(image_dir, img_name.replace('.png', '.dpt')), pred_depth * 10)
        plt.imsave(os.path.join(image_dir.replace('/raw','/cmap'), img_name), pred_depth, cmap='rainbow')
        cv2.imwrite(os.path.join(image_dir.replace('/raw','/rgb'), img_name), data['A_raw'].numpy().squeeze())

        gt_depth = gt[192:192+128,192:192+160] 
        pred_depth = pred_depth[192:192+128,192:192+160]
        ref_depth = np.percentile(gt_depth, 50)
        gt_mask = np.zeros_like(gt_depth)
        gt_mask[ref_depth>gt_depth] = 255
        init_mask = np.zeros_like(pred_depth)
        init_mask[ref_depth>pred_depth] = 255
        final_mask = sigmoid(10*(ref_depth-pred_depth)) * 255

        imsave(os.path.join(image_dir.replace('/raw','/occ_mask_gt'), img_name), gt_mask.astype(np.uint8))
        imsave(os.path.join(image_dir.replace('/raw','/occ_mask_init'), img_name), init_mask.astype(np.uint8))
        imsave(os.path.join(image_dir.replace('/raw','/occ_mask_final'), img_name), final_mask.astype(np.uint8))
