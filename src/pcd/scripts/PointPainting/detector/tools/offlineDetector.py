import sys,os,cv2
import time
PAIINTING_PATH = '../../painting'
sys.path.insert(0,PAIINTING_PATH)
DETECTOR_PATH = '../'
sys.path.insert(0,DETECTOR_PATH)

import argparse
from functools import partial

from pathlib import Path

import numpy as np
import torch


from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file

from lidar_msgs.msg import DetectedObject, DetectedObjectArray


from PointPainting.painting.my_painting import Painter
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
from pcdet.datasets.processor.data_processor import DataProcessor
import cv_bridge
from collections import defaultdict

def points_cam2img(points_3d, proj_mat):
    """Project points in camera coordinates to image coordinates.
    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.
    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = np.eye(4)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = np.concatenate((points_3d, np.ones(points_shape)), axis=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    return point_2d_res


def bbox_to_box(dimension, location, rot_y, init_mat):
    """
    boxes:
          4(000)              7(001)
          -------------------
         /|                 /|
  5(100)/ |         6(101) / |
        -------------------  |
        | |               |  |
        | |               |  |
        | |0(010)         |  |3(011)
        |  ------X--------|---
        | /               | /
        |/1(110)          |/
        -------------------2(111)
    connect line:
    ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
     (4, 5), (4, 7), (2, 6), (5, 6), (6, 7),
     (1, 6), (2, 5))
    """
    h, w, l = dimension
    x, y, z = location

    points = np.array(()).reshape(-1,3)
    # 010
    xyz = [x - l/2*np.cos(rot_y) - w/2*np.sin(rot_y), y + h*0   , z + l/2*np.sin(rot_y) - w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0)
    # 110
    xyz = [x + l/2*np.cos(rot_y) - w/2*np.sin(rot_y), y + h*0   , z - l/2*np.sin(rot_y) - w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0)
    # 111
    xyz = [x + l/2*np.cos(rot_y) + w/2*np.sin(rot_y), y + h*0   , z - l/2*np.sin(rot_y) + w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0) # [8, 3]
    # 011
    xyz = [x - l/2*np.cos(rot_y) + w/2*np.sin(rot_y), y + h*0   , z + l/2*np.sin(rot_y) + w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0)
    # 000
    xyz = [x - l/2*np.cos(rot_y) - w/2*np.sin(rot_y), y - h/1     , z + l/2*np.sin(rot_y) - w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0)
    # 100
    xyz = [x + l/2*np.cos(rot_y) - w/2*np.sin(rot_y), y - h/1     , z - l/2*np.sin(rot_y) - w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0)
    # 101
    xyz = [x + l/2*np.cos(rot_y) + w/2*np.sin(rot_y), y - h/1     , z - l/2*np.sin(rot_y) + w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0)
    # 001
    xyz = [x - l/2*np.cos(rot_y) + w/2*np.sin(rot_y), y - h/1     , z + l/2*np.sin(rot_y) + w/2*np.cos(rot_y)]
    xyz = np.array(xyz).reshape(-1,3)
    points = np.concatenate((points, xyz), axis=0)

    points = points_cam2img(points, init_mat) # [8, 2]
    return np.min(points[:,0]), np.min(points[:,1]), np.max(points[:,0]), np.max(points[:,1])


def get_quaternion_from_euler(roll=0, pitch=0, yaw=0):
    """
    Convert an Euler angle to a quaternion.

    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return [qx, qy, qz, qw]

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()

def collate_batch(batch_list, _unused=False):
    data_dict = defaultdict(list)
    for cur_sample in batch_list:
        for key, val in cur_sample.items():
            data_dict[key].append(val)
    batch_size = len(batch_list)
    ret = {}

    for key, val in data_dict.items():
        try:
            if key in ['voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = max([len(x) for x in val])
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d
            else:
                ret[key] = np.stack(val, axis=0)
        except:
            print('Error in collate_batch: key=%s' % key)
            raise TypeError

    ret['batch_size'] = batch_size
    return ret

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar_painted.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='../output/kitti_models/pointpillar_painted/default/ckpt/checkpoint_epoch_80.pth', help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--mode', type=str, default='file', help='')
    parser.add_argument('--lidar_dir', type=str, default='/media/johnhe/8C6DA37F246BD3C1/rosbag/zijian_2030demo/data/bin')
    parser.add_argument('--image_dir', type=str, default='/media/johnhe/8C6DA37F246BD3C1/rosbag/zijian_2030demo/data/img')
    parser.add_argument('--label_dir', type=str, default='/media/johnhe/8C6DA37F246BD3C1/rosbag/zijian_2030demo/data/txt')

    args = parser.parse_args()
   
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

class GLOBAL():
    receive_msg = 0


def offlineDetection(args, points, image):
    print('--------' + f"  {GLOBAL.receive_msg}  " + '--------')
    t0 = time.time()
    GLOBAL.receive_msg += 1

    model = args[0]
    painter:Painter  = args[1]
    bridge:cv_bridge.CvBridge = args[2]
    point_feature_encoder = args[3]
    data_processor = args[4]

    debug_dict = args[5]
    Tr_velo_to_cam = painter.calib['Tr_velo2cam'] # [4, 4]

    points = np.array(list(points))
    points = np.array(points, dtype=np.float32).reshape(-1, 4)
    points = points[~np.isnan(points).any(axis=1)]
    points[:,3] = points[:, 3] / points[:, 3].max()

    ### points prepared
    rectified_image = image
    # painting
    t0 = time.time()
    # points = debug_dict['points']
    points = painter.paint(points,rectified_image)
    print(f"painting cost time {time.time() - t0}")
    
    t0 = time.time()
    input_dict = {'points': points,'frame_id': 0}
    # prepare_data
    
    data_dict = point_feature_encoder.forward(input_dict)
    data_dict = data_processor.forward(data_dict=data_dict)
    data_dict.pop('gt_names', None)
    
    data_dict = collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    # pred
    pred_dicts, ret_dict = model(data_dict)
    print(f"predict cost time {time.time() - t0}")
    # pub msgs
    t0 = time.time()
    pred_dicts = pred_dicts[0]
    # print(pred_dicts['pred_labels'][-1].item())
    all_labels = ['Car', 'Pedestrian', 'Cyclist']
    objects = DetectedObjectArray()

    objects.header.frame_id = "rslidar"
    # print(objects.header.stamp.secs, objects.header.stamp.nsecs)
    results = []
    for i in range(len(pred_dicts['pred_labels'])):
        if (pred_dicts['pred_labels'][i].item() == 1 and pred_dicts['pred_scores'][i].item() > 0.4) or \
                (pred_dicts['pred_labels'][i].item() == 2 and pred_dicts['pred_scores'][i].item() > 0.4) or \
                (pred_dicts['pred_labels'][i].item() == 3 and pred_dicts['pred_scores'][i].item() > 0.4):
            detectedObject = DetectedObject()

            detectedObject.header.frame_id = "rslidar"
            detectedObject.label = all_labels[pred_dicts['pred_labels'][i].item() - 1]
            detectedObject.score = pred_dicts['pred_scores'][i].item()
            detectedObject.valid = pred_dicts['pred_scores'][i].item() > 0.5
            detectedObject.pose.position.x = pred_dicts['pred_boxes'][i][0].item()
            detectedObject.pose.position.y = pred_dicts['pred_boxes'][i][1].item()
            detectedObject.pose.position.z = pred_dicts['pred_boxes'][i][2].item()
            cam_xyz = Tr_velo_to_cam @ np.array([   pred_dicts['pred_boxes'][i][0].item(),
                                                    pred_dicts['pred_boxes'][i][1].item(),
                                                    pred_dicts['pred_boxes'][i][2].item(),
                                                    1
                                                    ]).reshape(4,1)
            pass
            detectedObject.pose_reliable = True
            yaw = pred_dicts['pred_boxes'][i][6].item() - 0*np.pi # to rotation_y

            detectedObject.pose.orientation.x, detectedObject.pose.orientation.y, detectedObject.pose.orientation.z, detectedObject.pose.orientation.w = get_quaternion_from_euler(0, 0, yaw)
            # lidar 
            detectedObject.dimensions.x = pred_dicts['pred_boxes'][i][3].item() # l
            detectedObject.dimensions.y = pred_dicts['pred_boxes'][i][4].item() # w
            detectedObject.dimensions.z = pred_dicts['pred_boxes'][i][5].item() # h        
            objects.objects.append(detectedObject)
            to_format = lambda x:f"{x:.2f}" if type(x)!=type(str()) else x
            dimension = (pred_dicts['pred_boxes'][i][5].item(),
                        pred_dicts['pred_boxes'][i][4].item(),
                        pred_dicts['pred_boxes'][i][3].item())
            location = (float(cam_xyz[0]),
                        float(cam_xyz[1]) + pred_dicts['pred_boxes'][i][5].item()/2,
                        float(cam_xyz[2]))
            box_2d = bbox_to_box(dimension, location, 3/2*np.pi - pred_dicts['pred_boxes'][i][6].item(), painter.calib['P2'])

            result = (  all_labels[pred_dicts['pred_labels'][i].item() - 1],
                        0,0,0,box_2d,
                        pred_dicts['pred_boxes'][i][5].item(),
                        pred_dicts['pred_boxes'][i][4].item(),
                        pred_dicts['pred_boxes'][i][3].item(),
                        float(cam_xyz[0]),
                        float(cam_xyz[1]) + pred_dicts['pred_boxes'][i][5].item()/2, # 底面中心点
                        float(cam_xyz[2]),
                        3/2*np.pi - pred_dicts['pred_boxes'][i][6].item())
            result_str = " ".join([to_format(item) for item in result])
            results.append(result_str)
        else:
            break

    
    print(f"postprocess cost time {time.time() - t0}")
    if len(objects.objects):
        return results
    else:
        return []



def realtime(args, cfg):
    dist_test = False


    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    dataset_cfg=cfg.DATA_CONFIG
    test_set = KittiDataset(
        dataset_cfg=dataset_cfg,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=False,
        logger=None,
    )

    # build_network
    from pcdet.models.detectors import build_detector
    from pcdet.models.detectors.pointpillar import PointPillar
    model = PointPillar(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model = build_detector(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # model = None
    SEG_NET = 1
    extrinsics_path = './calib.txt'
    painter = Painter(SEG_NET,calib_path=extrinsics_path)
    bridge  = cv_bridge.CvBridge()

    point_feature_encoder = PointFeatureEncoder(
            cfg.DATA_CONFIG.POINT_FEATURE_ENCODING,
            point_cloud_range=np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        )
    
    data_processor = DataProcessor(
            cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32),
            training=False
        )

    ## debug begin ##
    # codes for debugging
    # you can add anything useful for debugging to the debug_dict
    debug_dict = {}
    # import pickle
    # with open('points.pkl','rb') as f:
    #     debug_dict['points'] = pickle.load(f)
    ## debug end ##
    with torch.no_grad():
        model.load_params_from_file(filename=args.ckpt, logger=None, to_cpu=dist_test)
        model.cuda()
        model.eval()

        # callback arguments, this args will be binded by partial funtions.
        cb_args = [model,painter,bridge,point_feature_encoder,data_processor,debug_dict]
        offlineDetection_with_args = partial(offlineDetection,cb_args)
        img_dir = args.image_dir
        lidar_dir = args.lidar_dir
        labeb_dir = args.label_dir
        for idx in range(len(os.listdir(img_dir))):
            img_path = os.path.join(img_dir,f'{idx:06d}.png')
            lidar_path = os.path.join(lidar_dir,f'{idx:06d}.bin')
            label_path = os.path.join(labeb_dir,f'{idx:06d}.txt')
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            cv_image = cv2.imread(img_path)
            results_str_list = offlineDetection_with_args(points,cv_image)
            with open(label_path,'w') as f:
                f.write('\n'.join(results_str_list))
        
        



if __name__ == '__main__':
    args, cfg = parse_config()
    realtime(args, cfg)
    
