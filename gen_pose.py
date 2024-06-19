"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'


def check_input():

    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')



def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')



'''
args: Namespace(cfg='configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml', 
                checkpoint='pretrained_models/halpe26_fast_res50_256x192.pth', 
                sp=True, detector='yolo', detfile='', inputpath='examples/demo/', 
                inputlist='', inputimg='', outputpath='examples/res/', save_img=True, vis=False,
                 showbox=False, profile=False, format=None, min_box_area=0, detbatch=5, posebatch=64, 
                 eval=False, gpus=[0], qsize=1024, flip=False, debug=False, video='', webcam=-1, save_video=False, 
                 vis_fast=False, pose_flow=False, pose_track=False, device=device(type='cuda', index=0), tracking=False)

cfg:  {'DATASET': {'TRAIN': {'TYPE': 'Halpe_26', 'ROOT': './data/halpe/', 'IMG_PREFIX': 'images/train2015', 
        'ANN': 'annotations/halpe_train_v1.json', 'AUG': {'FLIP': True, 'ROT_FACTOR': 40, 'SCALE_FACTOR': 0.3, 
        'NUM_JOINTS_HALF_BODY': 11, 'PROB_HALF_BODY': -1}}, 'VAL': {'TYPE': 'Halpe_26', 'ROOT': './data/halpe/', 
        'IMG_PREFIX': 'images/val2017', 'ANN': 'annotations/halpe_val_v1.json'}, 
        'TEST': {'TYPE': 'Halpe_26_det', 'ROOT': './data/halpe/', 'IMG_PREFIX': 'images/val2017', 
        'DET_FILE': './exp/json/test_det_yolo.json', 'ANN': 'annotations/halpe_val_v1.json'}}, 
        'DATA_PRESET': {'TYPE': 'simple', 'SIGMA': 2, 'NUM_JOINTS': 26, 'IMAGE_SIZE': [256, 192], 'HEATMAP_SIZE': [64, 48]}, '
MODEL': {'TYPE': 'FastPose', 'PRETRAINED': '', 'TRY_LOAD': '', 'NUM_DECONV_FILTERS': [256, 256, 256], 'NUM_LAYERS': 50},
 'LOSS': {'TYPE': 'MSELoss'}, 'DETECTOR': {'NAME': 'yolo', 'CONFIG': 'detector/yolo/cfg/yolov3-spp.cfg', 
 'WEIGHTS': 'detector/yolo/data/yolov3-spp.weights', 'NMS_THRES': 0.6, 'CONFIDENCE': 0.05}, 
'TRAIN': {'WORLD_SIZE': 4, 'BATCH_SIZE': 48, 'BEGIN_EPOCH': 0, 'END_EPOCH': 200, 
'OPTIMIZER': 'adam', 'LR': 0.001, 'LR_FACTOR': 0.1, 'LR_STEP': [50, 70], 'DPG_MILESTONE': 90, 'DPG_STEP': [110, 130]}}

'''
import re
if __name__ == "__main__":

    mode = 'video'

    print("args:", args)
    print('cfg: ', cfg)

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)


    input_source= 'D:\\st-gcn\\data\\KTH\\training_lib_KTH_cut_6s\\jogging\\person01_jogging_d1_uncomp(003320-021320).avi'
    # input_source ='C:\\Users\\Administrator\\Desktop\\pyspace\\pyskl-main\\demo\\ntu_sample.avi'
    filename = re.split(r'\\',input_source)[-1] + '.json'
    cfg.filename = filename



    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=4, mode=mode, queueSize=1024)
    det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load('pretrained_models/halpe26_fast_res50_256x192.pth', map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {'dt': [], 'pt': [], 'pn': [] }

    # Init data writer
    queueSize = 1024

    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = 64

    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    hm_j = pose_model(inps_j)

                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)

                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()
        while(writer.running()):
            # time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C

        det_loader.terminate()
        while(writer.running()):
            # time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
        writer.stop()


