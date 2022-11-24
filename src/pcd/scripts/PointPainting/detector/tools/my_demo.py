import argparse
import glob
from pathlib import Path

# import mayavi.mlab as mlab
import numpy as np
import torch

import sys
sys.path[9] = '/home/johnhe/PointPaintingNode_ws/src/pcd/scripts/PointPainting/detector'
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import visualize_utils as V
import cv2
color_map = {1:(0, 255, 0),
            2:(0, 0, 255),
            3:(255, 0, 0)
            }
index_to_cls = {
    3:'Cyclist',
    1:'Car',
    2:'Pedestrian'
}
def draw_cuboid(image, qs, color=(0, 0, 255), thickness=1,cls_type='test'):
    ''' Draw 3d bounding box in image
      qs: (8,3) array of vertices for the 3d box in following order:
          1 -------- 0
          /|         /|
        2 -------- 3 .
        | |        | |
        . 5 -------- 4
        |/         |/
        6 -------- 7
      '''
    qs = qs.astype(np.int32)
    if not ((qs[:,0] < image.shape[1]).all() and (qs[:,1] < image.shape[0]).all()):
        return image
        
    for k in range(0, 4):
      i, j = k, (k + 1) % 4
      # use LINE_AA for opencv3
      cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]),
             color, thickness, cv2.LINE_AA)

      i, j = k + 4, (k + 1) % 4 + 4
      cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]),
             color, thickness, cv2.LINE_AA)

      i, j = k, k + 4
      cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]),
             color, thickness, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size = cv2.getTextSize(cls_type, font, 1, 2)
    start_point = sorted(qs,key=lambda x:(x[0],x[1]))[0]
    text_origin = np.array([start_point[0], start_point[1]])
    cv2.rectangle(image, tuple(text_origin), (text_origin[0] + int(label_size[0][0] * 0.7) ,text_origin[1] - label_size[0][1]),
                  color=color, thickness = -1) 
    cv2.putText(image, cls_type, (start_point[0], start_point[1] - 5), font, 0.7, (255, 255, 255), 2)
    
    return image


def get_calib_from_file(calib_file):
    """
    usage:
    calib['Tr_velo2cam']
    """
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
import os
root_path = '/media/johnhe/8C6DA37F246BD3C1/rosbag/zijian_2030demo/data/'
calib_files = os.listdir(root_path + 'calib/')
calib_files.sort()
img_files = os.listdir(root_path + 'image_2/')
img_files.sort()

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pointpillar_painted.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/kitti/training/painted_lidar/',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='../output/kitti_models/pointpillar_painted/default/ckpt/checkpoint_epoch_80.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_path', type=str, default='/media/johnhe/8C6DA37F246BD3C1/rosbag/zijian_2030demo/data/painted_lidar_img', help='specify the save path')
    parser.add_argument('--start_idx', type=int, default=0, help='specify the image index of the begining')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    # mlab.options.offscreen = True
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            print(idx)
            if idx < args.start_idx : continue
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            img = root_path + 'image_2/' + img_files[idx]
            img = cv2.imread(img)
            # @me
            for i in range(len(pred_dicts[0]['pred_labels'])):
                if (pred_dicts[0]['pred_labels'][i].item() == 1 and pred_dicts[0]['pred_scores'][i].item() > 0.4) or \
                        (pred_dicts[0]['pred_labels'][i].item() == 2 and pred_dicts[0]['pred_scores'][i].item() > 0.4) or \
                        (pred_dicts[0]['pred_labels'][i].item() == 3 and pred_dicts[0]['pred_scores'][i].item() > 0.4):

                    print(f"found idx of {idx}")
                    calib = root_path + 'calib/' + calib_files[idx]
                    calib = get_calib_from_file(calib)

                    get_square_mat = lambda calib34:np.vstack([calib34,[0,0,0,1]])
                    EXT = get_square_mat(calib['Tr_velo2cam'])
                    INT = get_square_mat(calib['P2'])

                    box_3d_lidar = boxes_to_corners_3d(pred_dicts[0]['pred_boxes'][i].unsqueeze(0))[0]
                    box_3d_4 = np.vstack([box_3d_lidar.cpu().T,[1,1,1,1,1,1,1,1]])
                    box_3d_4_img = INT @ EXT @ box_3d_4 
                    tmp_res = (box_3d_4_img / box_3d_4_img[2,:])[:2,:]

                    draw_cuboid(img,tmp_res.T,color=color_map[pred_dicts[0]['pred_labels'][i].item()],cls_type=index_to_cls[pred_dicts[0]['pred_labels'][i].item()])
            save_path = root_path + 'image_result_3d_to_2d/' + img_files[idx]
            cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(save_path)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # mlab.savefig(f"{args.save_path}/{idx:06d}.png")
            # mlab.clf()
            # mlab.close()
            # mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
