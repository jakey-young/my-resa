import copy
import os
import os.path as osp
import time
import shutil
import torch
import warnings
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import cv2
import numpy as np
import utils.transforms as tf
import argparse
from utils.config import Config
# from runner.runner import Runner
from datasets import build_dataloader
import time
import torch
import numpy as np
from tqdm import tqdm
import pytorch_warmup as warmup
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
from models.registry import build_net
from runner.registry import build_trainer, build_evaluator
from runner.optimizer import build_optimizer
from runner.scheduler import build_scheduler
from datasets import build_dataloader
from runner.recorder import build_recorder
from runner.net_utils import save_model, load_network
import time
warnings.simplefilter("always")

class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
        self.net, device_ids=range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.transform = self.transform_val()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        # self.evaluator = build_evaluator(self.cfg)
        self.warmup_scheduler = warmup.LinearWarmup(
            self.optimizer, warmup_period=5000)
        self.metric = 0.
        self.net.eval()

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from,
                     finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)


    def transform_val(self):
        val_transform = torchvision.transforms.Compose([
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return val_transform
    def validate(self, img0):

        img = img0[160:, :, :]
        img, = self.transform((img,))
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output = self.net(img)
            return self.evaluate(img0, output)


    def evaluate(self, img, output):
        seg_pred, exist_pred = output['seg'], output['exist']
        seg_pred = F.softmax(seg_pred, dim=1)
        seg_pred = seg_pred.detach().cpu().numpy()
        # exist_pred = exist_pred.detach().cpu().numpy()
        return self.evaluate_pred(img, seg_pred)

    def evaluate_pred(self, img, seg_pred):
        list2 = []
        list3 = []
        list4 = []
        point = {}
        formula = {}
        seg = seg_pred[0]
        # exist = [1 if exist_pred[b, i] >
        #               0.5 else 0 for i in range(self.cfg.num_classes - 1)]
        lane_coords = self.probmap2lane(seg,  thresh=0.6)
        list1 = copy.deepcopy(lane_coords)
        for j in range(len(lane_coords)):
            for i in range(len(lane_coords[j])):
                if lane_coords[j][i][0] == -1:
                    list1[j].remove(lane_coords[j][i])
        for j in range(len(list1)):
            for i in range(len(list1[j])):
                list2.append(list1[j][i][0])
                list3.append(list1[j][i][1])
                point['x' + str(j)] = list2
                point['y' + str(j)] = list3
            list2 = []
            list3 = []
        for j in range(len(list1)):
            y = np.array(point['y' + str(j)])
            y1 = -y
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            print(y)
            x = np.array(point['x' + str(j)])
            print(x)
            print(1111111111111111111111)
            z = np.polyfit(y1, x, 5)
            formula[str(j)] = z
        #print(img)
        for j in range(len(lane_coords)):
            for i in range(len(lane_coords[j])):
                if lane_coords[j][i][1] == 360:
                    list4.append(abs(640 - lane_coords[j][i][0]))
        a = list4.index(min(list4))
        list4[a] = 10000
        b = list4.index(min(list4))
        p1 = np.poly1d(formula[str(a)])
        p2 = np.poly1d(formula[str(b)])
        x2 = np.array(x_)
        x3 = -x2
        print("88855251")
        print(x3)
        y1 = p1(x3)
        y2 = p2(x3)
        y_ = (y1 + y2) / 2
        l = np.polyfit(x_, y_, 5)
        for i in range(len(x3) - 10):
            # cv2.line(img, (400, 710),
            #          (200, 160),
            #          (0, 255, 255), 3)
            # cv2.line(img, (0, 0),
            #          (800, 710),
            #          (0, 255, 255), 3)
            cv2.line(img, (int(y_[i]), int(x_[i])),
                     (int(y_[i+1]), int(x_[i+1])),
                     (0, 255, 255), 3)
            cv2.line(img, (int(y1[i]), int(x_[i])),
                     (int(y1[i+1]), int(x_[i+1])),
                     (255, 0, 255), 3)
            cv2.line(img, (int(y2[i]), int(x_[i])),
                     (int(y2[i+1]), int(x_[i+1])),
                     (255, 255, 0), 3)
        for j in range(len(lane_coords)):
            for i in range(len(lane_coords[j])):
                if lane_coords[j][i][0] == -1:
                    pass
                else:
                    try:
                        if lane_coords[j][i + 1][0] != -1:
                            cv2.line(img, (int(lane_coords[j][i][0]), int(lane_coords[j][i][1])),
                                     (int(lane_coords[j][i + 1][0]), int(lane_coords[j][i + 1][1])),
                                     (0, 0, 255), 3)
                        # matplotlib.use('TkAgg')
                        # # # a=A[0].cpu().detach().numpy()
                        # plt.imshow(img), plt.show()
                    except:
                        pass
        return img

    def probmap2lane(self, seg_pred,  resize_shape=(720, 1280), smooth=True, y_px_gap=10, pts=56, thresh=0.6):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (5, h, w)
        resize_shape:  reshape size target, (H, W)
        exist:       list of existence, e.g. [0, 1, 1, 0]
        smooth:      whether to smooth the probability or not
        y_px_gap:    y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold

        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if resize_shape is None:
            resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        _, h, w = seg_pred.shape
        H, W = resize_shape
        coordinates = []

        for i in range(self.cfg.num_classes - 1):
            prob_map = seg_pred[i + 1]
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = self.get_lane(prob_map, y_px_gap, pts, thresh, resize_shape)
            if self.is_short(coords):
                continue
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])

        if len(coordinates) == 0:
            coords = np.zeros(pts)
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
                 range(pts)])
        # print(coordinates)

        return coordinates

    def fix_gap(self, coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
            lane = coordinate[start:end + 1]
            if any(x < 0 for x in lane):
                gap_start = [i for i, x in enumerate(
                    lane[:-1]) if x > 0 and lane[i + 1] < 0]
                gap_end = [i + 1 for i,
                                     x in enumerate(lane[:-1]) if x < 0 and lane[i + 1] > 0]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (
                                    gap_end[i] - id) / gap_width * lane[gap_start[i]])
                if not all(x > 0 for x in lane):
                    print("Gaps still exist!")
                coordinate[start:end + 1] = lane
        return coordinate

    def is_short(self, lane):
        start = [i for i, x in enumerate(lane) if x > 0]
        if not start:
            return 1
        else:
            return 0

    def get_lane(self, prob_map, y_px_gap, pts, thresh, resize_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        resize_shape:  reshape size target, (H, W)

        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if resize_shape is None:
            resize_shape = prob_map.shape
        h, w = prob_map.shape
        H, W = resize_shape
        #H -= self.cfg.c
        H -= 160

        coords = np.zeros(pts)
        coords[:] = -1.0
        for i in range(pts):
            y = int((H - 10 - i * y_px_gap) * h / H)
            if y < 0:
                break
            line = prob_map[y, :]
            id = np.argmax(line)
            if line[id] > thresh:
                coords[i] = int(id / w * W)
        if (coords > 0).sum() < 2:
            coords = np.zeros(pts)
        self.fix_gap(coords)
        # print(coords.shape)

        return coords

class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        if not name.endswith('.mp4'):  # 保证文件名的后缀是.mp4
            name += '.mp4'
            warnings.warn('video name should ends with ".mp4"')
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width        # 宽
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频，编码需要为mp4v
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)

    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view

    cfg.work_dirs = args.work_dirs + '/' + cfg.dataset.train.type

    cudnn.benchmark = True
    cudnn.fastest = True

    runner = Runner(cfg)
    CAP = cv2.VideoCapture('1212.mp4')
    vw = VideoWriter('test22.mp4', 1024, 360)
    while True:
        OPEN, frame = CAP.read()
        img=copy.deepcopy(frame)
    # img = cv2.imread('/home/lgh/xiangm/dataset/clips/0313-1/60/1.jpg')
    # val_loader = build_dataloader(cfg.dataset.val, cfg, is_train=False)
        result = runner.validate(frame)
    # plt.imshow(result),plt.show()
        result=cv2.resize(result,(512,360))
        img=cv2.resize(img,(512,360))
        imgStackH = np.hstack((img, result))
        vw.write(imgStackH)
        cv2.imshow('result', imgStackH)
        #print(imgStackH.shape)
        # cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    vw.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',default='configs/tusimple.py',help='train config file path')
    parser.add_argument(
        '--work_dirs', type=str, default='work_dirs',
        help='work dirs')
    parser.add_argument(
        '--load_from',default='/home/ee615/YJQ/resa/tusimple_resnet34.pth',
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument(
        '--validate',default='/home/ee615/YJQ/resa/tusimple_resnet34.pth',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--view',default=True,
        action='store_true',
        help='whether to show visualization result')
    parser.add_argument('--gpus', nargs='+',  default='0')
    parser.add_argument('--seed', type=int,
                        default=None, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    color = [(255, 0, 0), (0, 255, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
             (138, 43, 226)]
    x_ = [710, 700, 690, 680, 670, 660, 650, 640, 630, 620, 610, 600, 590, 580, 570, 560, 550, 540, 530, 520, 510, 500,
          490, 480, 470, 460, 450, 440, 430, 420, 410, 400, 390, 380, 370, 360, 350, 340, 330, 320, 310, 300, 290, 280,
          270, 260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160]
    main()