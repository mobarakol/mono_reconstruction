from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import warnings
warnings.filterwarnings('ignore')

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

def compute_scale(gt, pred,min,max):
    mask = np.logical_and(gt > min, gt < max)
    pred = pred[mask]
    gt = gt[mask]
    scale = np.median(gt) / np.median(pred)
    return scale

def reconstruct_pointcloud(rgb, depth, cam_K, vis_rgbd=False):

    rgb = np.asarray(rgb, order="C")
    rgb_im = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
    if vis_rgbd:
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = cam_K
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        cam
    )

    return pcd
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    if opt.dataset == "endonasal":
            print('Preparing endonasal dataloader.')
            seqs_recons = ['reconstruction']
            dataset_recon = datasets.EndonasalDataset(
                opt.data_path, seqs_recons, opt.height, opt.width, [0], 4, is_train=False)
            
            num_train_samples = len(dataset_recon)
    else:
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "3d_reconstruction.txt"))
        dataset_recon = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                        encoder_dict['height'], encoder_dict['width'],
                                        [0], 4, is_train=False)

    save_dir = "reconstruction_scale"
    os.makedirs(save_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset_recon, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    print('sample size:', len(dataset_recon))
    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))
    
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    rgbs = []
    pred_disps = []
    cam_Ks = []
    inference_times = []
    filepath = []

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for data in tqdm(dataloader):
            print('batch_size', len(data))
            input_color = data[("color", 0, 0)].cuda()
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            time_start = time.time()
            output = depth_decoder(encoder(input_color))
            inference_time = time.time() - time_start
            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            rgbs.append(input_color)
            pred_disps.append(pred_disp)
            cam_Ks.append(data[("K", 0)])
            inference_times.append(inference_time)
            filepath.append(data['filepath'][0])

        
    pred_disps = np.concatenate(pred_disps)


    if opt.visualize_depth:
        vis_dir = os.path.join(opt.load_weights_folder, "vis_depth")
        os.makedirs(vis_dir, exist_ok=True)
        
    print("-> Reconstructing")

    pcds = []
    for i in tqdm(range(pred_disps.shape[0])):
        
        pred_disp = pred_disps[i]
        pred_depth = 1/pred_disp
        pred_height, pred_width = pred_depth.shape[:2]

        rgb = rgbs[i].squeeze().permute(1,2,0).cpu().numpy() * 255
        print('bef', rgb.max(), rgb.min(), pred_depth.max(), pred_depth.min())
        cam_K = cam_Ks[i][0,:3,:3].numpy()
        if opt.visualize_depth:
            vis_pred_depth = render_depth(pred_depth)
            print('filepath[i]:',filepath[i])
            vis_file_name = os.path.join(vis_dir, os.path.basename(filepath[i]))
            vis_pred_depth.save(vis_file_name)

        scale = 100
        print('scale', scale)
        pred_depth *= scale
        
        # pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        # pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        print('aft',rgb.max(), rgb.min(), pred_depth.max(), pred_depth.min())
        print('intrinsic', cam_K)
        pcd = reconstruct_pointcloud(rgb, pred_depth, cam_K, vis_rgbd=False)
        # o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)
    print('Saving point clouds...')
    for i, pcd in enumerate(pcds):
        fn = os.path.join(save_dir, os.path.basename(filepath[i])[:-4] + ".ply")
        o3d.io.write_point_cloud(fn, pcd)
    
    print('Point clouds saved to', save_dir)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
