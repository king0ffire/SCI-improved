import os
import sys
import time
import glob

import cv2
import numpy as np
import torch
import torchvision

import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from model import *
from multi_read_data import MemoryFriendlyLoader
import matlab.engine
import csv
from skimage.metrics import peak_signal_noise_ratio,structural_similarity


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=1000, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')   # 模块内迭代数
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Metric-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)
mat_path = args.save+'/mat_images/'
os.makedirs(mat_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("Test file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)


    model = Network(stage=args.stage) # 内模块3

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)


    #train_low_data_names = './train'
    #TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')  #[数据,名字]


    test_low_data_names = './data/lol'
    real_light_data_names=''
    logging.info("Metric test images:%s"%test_low_data_names)
    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    '''train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True)'''

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True)

    file="./EXP/Train-20221125-164748-52000/model_epochs/weights_100.pt"
    model.load_state_dict(torch.load(file))
    logging.info("Using model:%s" % file)

    eng=matlab.engine.start_matlab()
    #file = open(args.save+'/result.csv', mode="w", newline="")
    #csvf=csv.writer(file)
    #csvf.writerow(["name","","PSNR","","SSIM","","NIQE"])
    file = open(args.save + '/result.csv', mode="w", newline="")
    csvf = csv.writer(file)
    csvf.writerow(["name", "", "PSNR", "", "SSIM"])
    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('\\')[-1].split('.')[0]
            illu_list, ref_list, input_list, atten = model(input)
            print(model._loss(input))

            #a=input[0].cpu().float().numpy()
            #b=ref_list[0][0].cpu().float().numpy()
            #a=np.clip(a*255.0,0,255.0).astype('uint8').transpose(1,2,0)
            #b=np.clip(b*255.0,0,255.0).astype('uint8').transpose(1,2,0) #type: numpy.ndarray
            #b_mat=matlab.uint8(list(b.tolist()))
            #eng.imshow(b_mat)
            #niqe = eng.niqe(b_mat,nargout=1)
            #matim= Image.fromarray(b)
            #matim.save(mat_path+'/' + '%s.png' % (image_name))

            #c=torch.clamp(input*255.0,0,255.0).type(dtype=torch.uint8).type(dtype=torch.float32)
            #d=torch.clamp(ref_list[0]*255.0,0,255.0).type(dtype=torch.uint8).type(dtype=torch.float32)
            #psnr=PeakSignalNoiseRatio().__call__(c,d).item()
            #print("sys message psnr:"+str(psnr))
            #ssim=StructuralSimilarityIndexMeasure(data_range=1.0)
            #ssim=ssim(c,d).item()
            #print("sys message ssim:"+str(ssim))

            #logging.info('img:%s    PSNR:%.4f   SSIM:%.4f  NIQE:%.4f' % (image_name, psnr, ssim,niqe))
            #csvf.writerow([image_name,"",psnr,"",ssim,"",niqe])

            u_name = '%s.png' % (image_name)
            u_path = image_path + '/' + u_name
            save_images(ref_list[0], u_path)
        print("==> Start testing")
        tStart = time.time()
        trans = torchvision.transforms.ToTensor()
        channel_swap = (1, 2, 0)
        testfolder1 = "./data/lol_real"
        testfolder2 = image_path
        # test_est_folder = "outputs/eopch_%s_%04d/" % (t, epoch)
        # test_est_folder = opt.save + '/image_epochs/'+ str(epoch)
        tlist1 = [os.path.join(testfolder1, x) for x in sorted(os.listdir(testfolder1)) if is_image_file(x)]
        tlist2 = [os.path.join(testfolder2, x) for x in sorted(os.listdir(testfolder2)) if is_image_file(x)]
        '''est_list=[]
        for x in sorted(listdir(test_LL_folder)):
            if is_image_file(x):
                x=join(str(epoch)+"/",x)
                est_list.append(join(test_est_folder, x))'''


        for i in range(tlist1.__len__()):
            gt = cv2.imread(tlist1[i])
            est = cv2.imread(tlist2[i])
            psnr_val = peak_signal_noise_ratio(gt, est, data_range=255)
            ssim_val = structural_similarity(gt, est, multichannel=True)
            print("{}:psnr:{:.4f}    ssim:{:.4f}".format(os.path.join(tlist1[i]), psnr_val, ssim_val))
            csvf.writerow([os.path.basename(tlist1[i]), "", psnr_val, "", ssim_val])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

def metric():
    print("==> Start testing")
    tStart = time.time()
    trans = torchvision.transforms.ToTensor()
    channel_swap = (1, 2, 0)
    testfolder1 = "./data/lol_real"
    testfolder2 = "./EXP/Metric-20230128-174307/image_epochs"
    #test_est_folder = "outputs/eopch_%s_%04d/" % (t, epoch)
    #test_est_folder = opt.save + '/image_epochs/'+ str(epoch)
    tlist1 = [os.path.join(testfolder1, x) for x in sorted(os.listdir(testfolder1)) if is_image_file(x)]
    tlist2 = [os.path.join(testfolder2, x) for x in sorted(os.listdir(testfolder2)) if is_image_file(x)]
    '''est_list=[]
    for x in sorted(listdir(test_LL_folder)):
        if is_image_file(x):
            x=join(str(epoch)+"/",x)
            est_list.append(join(test_est_folder, x))'''
    file = open(args.save + '/result.csv', mode="w", newline="")
    csvf = csv.writer(file)
    csvf.writerow(["name", "", "PSNR", "", "SSIM"])

    for i in range(tlist1.__len__()):
        gt = cv2.imread(tlist1[i])
        est = cv2.imread(tlist2[i])
        psnr_val = peak_signal_noise_ratio(gt, est, data_range=255)
        ssim_val = structural_similarity(gt, est, multichannel=True)
        print("{}:psnr:{:.4f}    ssim:{:.4f}".format(os.path.join(tlist1[i]),psnr_val,ssim_val))
        csvf.writerow([os.path.basename(tlist1[i]), "", psnr_val, "", ssim_val])
main()