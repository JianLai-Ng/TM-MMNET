
import math
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
import argparse, random
from functools import partial
from CA_Block import resnet18_pos_attention
from PC_module import VisionTransformer_POS
from torch.utils.data import ConcatDataset
from torchvision.transforms import Resize, functional
from itertools import chain
from math import inf
import random
import boto3
import s3fs
import matplotlib.pyplot as plt 
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


fs = s3fs.S3FileSystem()


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name_train='training' #s3://2022-mer-s3bucket/Dataset/

training_path = os.path.join(input_path, channel_name_train)
result_path = model_path


#ml.c5.9xlarge
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plateau_epoch', type=float, default=100) #51
    parser.add_argument('--channeltype', type=str, default='r') #0.9
    parser.add_argument('--gamma', type=float, default=0.987) #0.9
    parser.add_argument('--part', type=str, default='26')
    parser.add_argument('--data_subdir', type=str, default= training_path, help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=36, help='Batch size.') #as per biggest loso subject 
    # parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0008, help='Initial learning rate for sgd.')
    #parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=0, help='Total training epochs.') #100
    #parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    # Data, model, and output directories
    #parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default='s3://{}/{}'.format('2022-mer-s3bucket', 'models/'))
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    return parser.parse_args()

# For iterable-style datasets, since each worker process gets a replica of the dataset object, naive multi-process loading will often result in duplicated data. Using torch.utils.data.get_worker_info() and/or worker_init_fn, users may configure each replica independently. (See IterableDataset documentations for how to achieve this. ) For similar reasons, in multi-process loading, the drop_last argument drops the last non-full batch of each worker’s iterable-style dataset replica.


class RafDataSet(data.Dataset):
    def __init__(self, data_subdir, phase,num_loso, channeltype, basic_aug = False):
        self.phase = phase
        self.transform = None
        self.data_subdir = data_subdir +'/'
        self.transform_norm = None
        self.i_crop = 0
        self.j_crop = 0
        self.channeltype = channeltype
        SUBJECT_COLUMN =0
        NAME_COLUMN = 1
        MIN_COLUMN = 2
        APEX_COLUMN = 3
        MAX_COLUMN = 4#

        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 6

        self.aug_folder = 'denoised_masked'
        self.image_prefix = 'md_reg_img'

        df = pd.read_csv(os.path.join(self.data_subdir, 'CASME2-coding-2023apr16.csv'))
        print(os.path.join(self.data_subdir, 'CASME2-coding-2023apr16.csv'))
        df['Subject'] = df['Subject'].apply(str)
        df = df[df.columns[[0,1,7,3,8,5,6]]]

        if phase == 'train':
            dataset = df.loc[df['Subject']!=str(num_loso)]
        else:
            dataset = df.loc[df['Subject'] == str(num_loso)]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        Onset_num = dataset.iloc[:, MIN_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, MAX_COLUMN].values
        Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.label_au = []
        self.sub= []
        self.file_names =[]
        a=0
        b=0
        c=0
        d=0
        e=0
        # use aligned images for training/testing
        for (f,sub,onset,apex,offset,label_all,label_au) in zip(File_names,Subject,Onset_num,Apex_num,Offset_num,Label_all,Label_au):


            if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'others':

                self.file_paths_on.append(onset)
                self.file_paths_off.append(offset)
                self.file_paths_apex.append(apex)
                self.sub.append(sub)
                self.file_names.append(f)
                if label_all == 'happiness':
                    self.label_all.append(0)
                    a=a+1
                elif label_all == 'repression':
                    self.label_all.append(1)
                    b=b+1
                elif label_all == 'disgust':
                    self.label_all.append(2)
                    c=c+1
                elif label_all == 'surprise':
                    self.label_all.append(3)
                    d=d+1
                else:
                    self.label_all.append(4)
                    e=e+1

            # label_au =label_au.split("+")
                if isinstance(label_au, int):
                    self.label_au.append([label_au])
                else:
                    label_au = label_au.split("+")
                    self.label_au.append(label_au)


        flatten_list = list(chain.from_iterable(self.label_au))
        au_set= list(set([str(a) for a in flatten_list]))
        au_set.sort()
        self.embedding_au_size = len(au_set)
        self.au_set_to_index = {}
        index = 0
        for an_ele in au_set:
            self.au_set_to_index[an_ele] = index 
            index+=1
        



            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
    def compute_optflow_magsincos(self, prev_image, current_image):
        t = self.channeltype
        
        old_shape = current_image.shape
        prev_image_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        current_image_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        assert current_image.shape == old_shape
        flow = None
        flow = cv2.calcOpticalFlowFarneback(prev=prev_image_gray,
                                            next=current_image_gray, flow=flow,
                                            pyr_scale=0.4, levels=1, winsize=10,
                                            iterations=3, poly_n=5, poly_sigma=1.2,
                                            flags=10)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag[mag == -inf] = 0
        mag[mag == inf] = 0

        # hsv = np.zeros(old_shape, dtype=np.uint8)
        # hsv[..., 1] = 255

        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        horz = flow[...,0]
        vert = flow[...,1]

        if t=='n':#5n

            new_min, new_max = -1, 1

            #HORZ
            horz_min, horz_max = horz.min(), horz.max()
            horz_normalized = (horz - horz_min)/(horz_max - horz_min)*(new_max - new_min) + new_min

            #VERT
            vert_min, vert_max = vert.min(), vert.max()
            vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min

            out = np.zeros((224,224,2), dtype = float)
            out[..., 0] = horz_normalized
            out[..., 1] = vert_normalized

        if t =='r':
            out = np.zeros((224,224,2), dtype = float)
            out[..., 0] = horz
            out[..., 1] = vert

        if t=='s':
            scaler = StandardScaler()
            horz_scaled = scaler.fit_transform(horz)
            vert_scaled = scaler.fit_transform(vert)
            out = np.zeros((224,224,2), dtype = float)
            out[..., 0] = horz_scaled
            out[..., 1] = vert_scaled

        if t=='nr': #V5
            new_min, new_max = -1, 1

            #HORZ
            horz_min, horz_max = horz.min(), horz.max()
            horz_normalized = (horz - horz_min)/(horz_max - horz_min)*(new_max - new_min) + new_min

            #VERT
            vert_min, vert_max = vert.min(), vert.max()
            vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
            out = np.zeros((224,224,4), dtype = float)
            out[..., 0] = horz_normalized
            out[..., 1] = horz
            out[..., 2] = vert
            out[..., 3] = vert_normalized

        if t=='rs': 
            scaler = StandardScaler()
            horz_scaled = scaler.fit_transform(horz)
            vert_scaled = scaler.fit_transform(vert)
            out = np.zeros((224,224,4), dtype = float)
            out[..., 0] = horz
            out[..., 1] = horz_scaled
            out[..., 2] = vert_scaled
            out[..., 3] = vert

        if t=='ns': #5b
            new_min, new_max = -1, 1

            #HORZ
            horz_min, horz_max = horz.min(), horz.max()
            horz_normalized = (horz - horz_min)/(horz_max - horz_min)*(new_max - new_min) + new_min

            #VERT
            vert_min, vert_max = vert.min(), vert.max()
            vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
            
            scaler = StandardScaler()
            horz_scaled = scaler.fit_transform(horz)
            vert_scaled = scaler.fit_transform(vert)
            out = np.zeros((224,224,4), dtype = float)
            out[..., 0] = horz_normalized
            out[..., 1] = horz_scaled
            out[..., 2] = vert_scaled
            out[..., 3] = vert_normalized    

        if t=='nrs':

            new_min, new_max = -1, 1

            #HORZ
            horz_min, horz_max = horz.min(), horz.max()
            horz_normalized = (horz - horz_min)/(horz_max - horz_min)*(new_max - new_min) + new_min

            #VERT
            vert_min, vert_max = vert.min(), vert.max()
            vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
            
            scaler = StandardScaler()
            horz_scaled = scaler.fit_transform(horz)
            vert_scaled = scaler.fit_transform(vert)
            out = np.zeros((224,224,6), dtype = float)
            out[..., 0] = horz
            out[..., 1] = horz_normalized
            out[..., 2] = horz_scaled
            out[..., 3] = vert_scaled
            out[..., 4] = vert_normalized    
            out[..., 5] = vert 

        return out

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = int(self.file_paths_on[idx])
            #apex = int(self.file_paths_apex[idx])
            offset = int(self.file_paths_off[idx])

            #mid_frame to be middle of onset and offset
            mid = int((onset+ offset)/2)
            #print(onset, mid, offset)

            on0 = int(random.randint(int(onset), int(onset + int(0.15* (mid - onset) / 4))))
            off0 = int(random.randint(int(offset - int(0.15* (offset - mid) / 4)), int(offset)))
            mid0 = int((on0+off0)/2)

            onmid0 = int((on0+ mid0)/2)
            midoff0 = int((off0+ mid0)/2)
                

            
            on0 = str(on0)
            onmid0 = str(onmid0)
            mid0 = str(mid0)
            midoff0 = str(midoff0)
            off0 = str(off0)

            sub ='sub' + str('%02d' % int(self.sub[idx]))
            fnn = str(self.file_names[idx])
            #print('frames decided')


        else:##sampling strategy for testing set
            on0 = int(self.file_paths_on[idx])
            #apex0 = int(self.file_paths_apex[idx])
            off0 = int(self.file_paths_off[idx])

            mid0 = int((on0+ off0)/2)
            onmid0 = int((on0+ mid0)/2)
            midoff0 = int((off0+ mid0)/2)

            on0 = str(on0)
            onmid0 = str(onmid0)
            mid0 = str(mid0)
            midoff0 = str(midoff0)
            off0 = str(off0)


            sub ='sub' + str('%02d' % int(self.sub[idx]))
            fnn= str(self.file_names[idx])


        aug_folder = self.aug_folder
        image_prefix = self.image_prefix

        pre_path = self.data_subdir +aug_folder+ '/'+ sub + '/'+ fnn +'/'+ image_prefix

        image_on0 = cv2.imread(pre_path+str(on0)+'.jpg')
        image_onmid0 = cv2.imread(pre_path+str(onmid0)+'.jpg')
        image_mid0 = cv2.imread(pre_path+str(mid0)+'.jpg')
        image_midoff0= cv2.imread(pre_path+str(midoff0)+'.jpg')
        image_off0= cv2.imread(pre_path+str(off0)+'.jpg')


        

        image_on0 = image_on0[:, :, ::-1] # BGR to RGB    
        image_onmid0 = image_onmid0[:, :, ::-1]
        image_mid0 = image_mid0[:, :, ::-1]
        image_midoff0 = image_midoff0[:, :, ::-1]
        image_off0 = image_off0[:, :, ::-1]
        #print(pre_path)

        # on0_dir =pre_path+str(on0)+'.jpg'
        # #print(on0_dir)
        # with fs.open(on0_dir, 'rb') as f:
        #     image_on0 = plt.imread(f, format='jpg') #already in RGB

        # onmid0_dir =pre_path+str(onmid0 )+'.jpg'
        # with fs.open(onmid0_dir, 'rb') as f:
        #     image_onmid0 = plt.imread(f, format='jpg') #already in RGB

        # with fs.open(on0_dir, 'rb') as f:
        #     image_mid0= plt.imread(f, format='jpg') #already in RGB

        # midoff0_dir = pre_path+str(midoff0)+'.jpg'
        # with fs.open(midoff0_dir, 'rb') as f:
        #     image_midoff0 = plt.imread(f, format='jpg') #already in RGB

        # off0_dir =pre_path+str(off0)+'.jpg'
        # with fs.open(off0_dir, 'rb') as f:
        #     image_off0= plt.imread(f, format='jpg') #already in RGB

        label_all = self.label_all[idx]
        label_au = self.label_au[idx]
        
        if aug_folder == 'masked_denoised':
            ## normalization for testing and training
            normalisation_transformation = transforms.Compose([
                        transforms.ToPILImage(),
                        #transforms.Resize((224, 224)), #already done
                        #transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                     std=[0.229, 0.224, 0.225]),
                                            ])
        else:
             ## normalization for testing and training
            normalisation_transformation = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)), #already done
                        #transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                     std=[0.229, 0.224, 0.225]),
                                            ])           

        image_on0 = normalisation_transformation(image_on0)
        image_onmid0 = normalisation_transformation(image_onmid0)
        image_mid0 = normalisation_transformation(image_mid0)
        image_midoff0 = normalisation_transformation(image_midoff0)
        image_off0 = normalisation_transformation(image_off0)


        ## data augmentation for training only
        if self.phase == 'train':
            
            #initiate sample standard
            self.crop_standard = 15
            self.flip_bool = random.randint(0,1)
            #self.rot_deg = random.randint(-3,3)
            #self.padcrop_or_cropinterpolate = random.randint(0,1)
            #self.padcrop_indices = transforms.RandomCrop(size = (224, 224)).get_params(transforms.Pad(self.crop_standard )(image_on0) , output_size=(224, 224))
            #self.padcrop_indices_i, self.padcrop_indices_j, self.padcrop_indices_h, self.padcrop_indices_w = self.padcrop_indices
            self.cropinterpolate_indices = transforms.RandomCrop(size = (224-self.crop_standard, 224)).get_params(
                            transforms.Pad(0)(image_on0) , output_size=(224-self.crop_standard, 224))
            self.cropinterpolate_indices_i, self.cropinterpolate_indices_j, self.cropinterpolate_indices_h, self.cropinterpolate_indices_w = self.cropinterpolate_indices

                    
            #pad and crop or crop and interpolate
            #ALL = [image_on0, image_onap0, image_apex0, image_apoff0, image_off0]
            ALL = [image_on0, image_onmid0, image_mid0, image_midoff0, image_off0]
            ALL_post = []

            for animage in ALL:
                animage = transforms.RandomHorizontalFlip(p=self.flip_bool)(animage) #flip
                #animage = transforms.functional.rotate(animage, angle = self.rot_deg) #rotate
                # if self.padcrop_or_cropinterpolate:
                #     #padcrop
                #     animage = transforms.Pad(self.crop_standard)(animage)
                #     animage = transforms.functional.crop(animage, self.padcrop_indices_i, self.padcrop_indices_j, self.padcrop_indices_h, self.padcrop_indices_w )
                # else:

                #crop vertically and interpolate
                animage = transforms.functional.crop(animage, self.cropinterpolate_indices_i, self.cropinterpolate_indices_j, self.cropinterpolate_indices_h, self.cropinterpolate_indices_w )
                
                #crop horizontally and interpolate
                hori_crop_aside = random.randint(0,int(self.crop_standard/2))
                animage = transforms.functional.crop(animage, 0, hori_crop_aside, 224, 224 - (2*hori_crop_aside) )

                animage = transforms.Resize((224, 224))(animage)
                
                ALL_post.append(animage)

            #image_on0, image_onap0, image_apex0, image_apoff0, image_off0 = ALL_post
            image_on0, image_onmid0, image_mid0, image_midoff0, image_off0 = ALL_post

   

        temp = torch.zeros(len(self.au_set_to_index))
        for an_au in label_au:
            temp[self.au_set_to_index[str(an_au)]] = 1

        #get optical flow map
        #print(type(np.asarray(image_on0)))
        of1 = self.compute_optflow_magsincos(np.asarray(image_on0), np.asarray(image_onmid0))
        of2 = self.compute_optflow_magsincos(np.asarray(image_onmid0), np.asarray(image_mid0))
        of3 = self.compute_optflow_magsincos(np.asarray(image_mid0), np.asarray(image_midoff0))
        of4 = self.compute_optflow_magsincos(np.asarray(image_midoff0), np.asarray(image_off0))
       


        return transforms.ToTensor()(image_on0),transforms.ToTensor()(of1), transforms.ToTensor()(of2), transforms.ToTensor()(of3), transforms.ToTensor()(of4),label_all, temp

def initialize_weight_goog(m, n=''): #Adjusted
    if isinstance(m, nn.Conv3d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def criterion2(y_pred, y_true): #no need adjust
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)


class MMNet(nn.Module): #ADJUSTED
    def __init__(self, initial_inchannels, embedding_au_size = None):
        super(MMNet, self).__init__()

        self.embedding_au_size = embedding_au_size

        self.conv_act = nn.Sequential(

            nn.Conv3d(in_channels=initial_inchannels, out_channels=180, kernel_size=(1,3,3), stride=(1,2,2),padding=(0,1,1), bias=False,groups=1),
            nn.BatchNorm3d(180),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=180, out_channels=512, kernel_size=(4,1,1), stride=(1,1,1),padding=(0,0,0), bias=False,groups=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            )

        ##Position Calibration Module(subbranch)
        self.vit_pos=VisionTransformer_POS(img_size=14,
        patch_size=1, embed_dim=2048, depth=2, num_heads=4, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.)
        self.resize=Resize([14,14])
        ##main branch consisting of CA blocks
        self.main_branch =resnet18_pos_attention()

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1 * 112 *112, 38,bias=False),

        )

        self.timeembed = nn.Parameter(torch.zeros(1, 4, 111, 111))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, onset, x1, x2, x3, x4, if_shuffle):
        ##onset:x1 apex:x5
        B = onset.shape[0]
        act = torch.stack([x1,x2,x3,x4], dim=2).float()
        #print('tensor shape:', act.shape)
        
        act = self.conv_act(act)
        act = torch.squeeze(act)
        # Position Calibration Module (subbranch)
        POS =self.vit_pos(self.resize(onset)).transpose(1,2).view(B,2048,14,14)

        #act =x5 -x1
       
  
        # main branch and fusion
        out,_, embedding =self.main_branch(act,POS, self.embedding_au_size)

        return out, embedding





def run_training(args):

    print('V5'+args.channeltype)
    criterion = torch.nn.CrossEntropyLoss()


    if 'hardest' in args.part :
        if '_' not in args.part:
            LOSO = [5,16,7,3,6,17,20,13,26,1]
        else:
            top_n = int(args.part.split('_')[-1])
            LOSO = [5,16,7,3,6,17,20,13,26,1][:top_n]
            
    elif 'h7-balance1' == args.part:
        LOSO = [1, 2, 4, 8, 9, 10, 11] 
    elif 'h7-balance2' == args.part:
        LOSO = [12, 13, 14, 15, 18, 19] 
    elif 'h7-balance3' == args.part:
        LOSO = [21, 22, 23, 24, 25, 26] 
        
    elif '[' in args.part :
        LOSO = eval(args.part)
    else:
        part= args.part.split('-')
        if len(part) == 2:
            start_loso = part[0]
            end_loso = part[-1]

            LOSO = [ap for ap in range(int(start_loso), int(end_loso)+1)]
        else:
            LOSO = [int(part),]

    print('Input parameter:',args.part)
    print('Subjects:',LOSO)
    print('Epochs:',args.epochs)
    
    #[subj, i, best_f1_epoch, val_dataset.__len__(), str(max_pos_label.tolist()), str(max_pos_pred.tolist()), str(max_TP.tolist()), subject_best_f1]
    testing_metrics_df = pd.DataFrame(columns = ['subject','total_trained_epochs', 'best_epoch', 'val_dataset_len', 'truth' ,'prediction', 'max_TP',
                                                 'best_FP', 'best_FN', 'best_accuracy_under_best_macro_f1', 'best_macro_f1'])





    for subj in LOSO:
        subj_result_df = pd.DataFrame(columns = ['Subject', 'Epoch','LR', 'Train_Loss','Train_Accuracy','Val_Loss', 'truth' ,'prediction','Val_Accuracy','Val_F1'])

        train_dataset = RafDataSet(args.data_subdir, phase='train', num_loso=subj, channeltype=args.channeltype, basic_aug=True)
        val_dataset = RafDataSet(args.data_subdir, phase='test', num_loso=subj, channeltype=args.channeltype,)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True, drop_last = False)
        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())
        


        max_corr = 0
        max_f1 = -1
        max_f1_best_acc = -1
        max_pos_pred = torch.zeros(5)
        max_pos_label = torch.zeros(5)
        max_TP = torch.zeros(5)
        ##model initialization
        initial_inchannels = len(args.channeltype) *2
        net_all = MMNet(initial_inchannels)

        params_all = net_all.parameters()

        # if args.optimizer == 'adam':
        optimizer_all = torch.optim.AdamW(params_all, lr=args.lr, weight_decay=0.6)
            ##optimizer for MMNet

        # elif args.optimizer == 'sgd':
        #     optimizer = torch.optim.SGD(params, args.lr,
        #                                 momentum=args.momentum,
        #                                 weight_decay=1e-4)
        # else:
        #     raise ValueError("Optimizer not supported.")
        ##lr_decay
        scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=args.gamma)

        net_all = net_all#.cuda()

        for i in range(1, int(args.epochs)+1):
            epoch_lr = scheduler_all.get_last_lr()[0]
            running_loss = 0.0
            correct_sum = 0
            running_loss_MASK = 0.0
            correct_sum_MASK = 0
            iter_cnt = 0

            net_all.train()


            for batch_i, (onset, of1, of2, of3,  of4, label_all,
            label_au) in enumerate(train_loader):
                batch_sz = of1.size(0)
                b, c, h, w = of1.shape
                iter_cnt += 1
                
                
                onset = onset#.cuda()
                of1 = of1#.cuda()
                of2 = of2#.cuda()
                of3 = of3#.cuda()
                of4 = of4#.cuda()

                label_all = label_all#.cuda()
                label_au = label_au#.cuda()

                ##train MMNet
                ALL, emb_au = net_all(onset, of1, of2, of3, of4, False)

                loss_all = criterion(ALL, label_all)

                optimizer_all.zero_grad()

                loss_all.backward()

                optimizer_all.step()
                running_loss += loss_all
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, label_all).sum()
                correct_sum += correct_num


                print('        iter:' +str(iter_cnt))
                #print('        mem allocated {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))
                print('        time:' + datetime.now().strftime("%d-%m-%Y, %H:%M:%S"))
                print('        loss:' +str(loss_all))

            

            ## lr decay
            if i <= args.plateau_epoch+1:

                scheduler_all.step()
            if i>=0:
                acc = correct_sum.float() / float(train_dataset.__len__())

                running_loss = running_loss / iter_cnt

                print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))


            train_loss = running_loss
            train_acc = acc
            
            FP = torch.zeros(5)
            FN = torch.zeros(5)
            TP = torch.zeros(5)



            ALL_CAT = None
            label_all_CAT = None

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                pre_lab_all = []
                Y_test_all = []
                net_all.eval()
                # net_au.eval()
                for batch_i, (
                onset, of1, of2, of3, of4, label_all, label_au) in enumerate(val_loader):
                    batch_sz = of1.size(0)
                    print('testing size: ',batch_sz)
                    b, c, h, w = of1.shape
                    
                    onset = onset#.cuda()
                    of1 = of1#.cuda()
                    of2 = of2#.cuda()
                    of3 = of3#.cuda()
                    of4 = of4#.cuda()

                    label_all = label_all#.cuda()
                    label_au = label_au#.cuda()
                    

                    ##test
                    ALL, embed_au = net_all(onset, of1, of2, of3, of4, False)

                    if ALL_CAT == None:
                        ALL_CAT = ALL
                    else:
                        ALL_CAT = torch.cat((ALL_CAT, ALL), 0)
                        
                    if label_all_CAT == None:
                        label_all_CAT = label_all
                    else:
                        label_all_CAT = torch.cat((label_all_CAT, label_all), 0)
                        
                loss = criterion(ALL_CAT, label_all_CAT)
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(ALL_CAT, 1) #returns chance, prediction  tensor([1, 4, 0, 4, 0, 0]) eg.
                correct_num = torch.eq(predicts, label_all_CAT) #tensor([False,  True,  True, False,  True, True]) eg.
                bingo_cnt += correct_num.sum().cpu() #3 eg.
                sample_cnt += ALL_CAT.size(0) #6 eg.
                

                print('PREDICTS:')
                print(predicts)
                print('TRUTH:')
                print(label_all_CAT)
                
                for a_class in range(5):

                    for elementp, elementl in zip(predicts, label_all_CAT):
                        if elementp == elementl and elementp == a_class:
                            TP[a_class] = TP[a_class] + 1
                        if elementp != elementl and elementp == a_class: #predict as class but not, false positive
                            FP[a_class] = FP[a_class] + 1
                        if elementp != elementl and elementl == a_class: #predict as not class is, false negative
                            FN[a_class] = FN[a_class] + 1

                            
                f1_list = []
                for a_class in range(5):
                    try:
                        f1_this_class = (2 * TP[a_class]) / ((2 * TP[a_class]) + FP[a_class] + FN[a_class])
                        f1_list.append(f1_this_class)
                    except:
                        print('0 base, not added to list')


                AVG_F1_visual = torch.mean(torch.stack(f1_list)).tolist()
                AVG_F1 = f1_score(label_all_CAT, predicts, average='macro')
                


                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)  
    
                acc = np.around(acc.detach().numpy(), 4)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt

                if AVG_F1 == max_f1:
                    if max_f1_best_acc < acc:
                        best_f1_epoch = i
                        max_f1 = AVG_F1
                        max_f1_best_acc = acc
                        max_pos_label = label_all_CAT
                        max_pos_pred = predicts
                        max_TP = TP
                        best_FN = FN
                        best_FP = FP
                        print('NEW ACC wtih Best Val F1 for subj'+str(subj) +':',str(max_f1)+', '+str(acc)+ 'at epoch:'+str(best_f1_epoch))
                        #save model
                        
                        with open(os.path.join(model_path, str(subj)+'model.pth'), 'wb') as f:
                            torch.save(net_all.state_dict(), f)
                        print('MODEL SAVED AS' +  str(subj)+'model.pth')

                elif AVG_F1 > max_f1:
                    best_f1_epoch = i
                    max_f1 = AVG_F1
                    max_f1_best_acc = acc
                    max_pos_label = label_all_CAT
                    max_pos_pred = predicts
                    max_TP = TP
                    best_FN = FN
                    best_FP = FP
                    print('NEW Best Val F1 for subj'+str(subj) +':',str(max_f1)+', at epoch:'+str(best_f1_epoch))
                    #save model
                    
                    with open(os.path.join(model_path, str(subj)+'model.pth'), 'wb') as f:
                        torch.save(net_all.state_dict(), f)
                    print('MODEL SAVED AS' +  str(subj)+'model.pth')
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, sklearn_F1-score:%.3f" % (i, acc, running_loss,  AVG_F1))
            
            #CHECK DATATYPES
                #subj_result_df = pd.DataFrame(columns = ['Subject', 'Epoch','LR', 'Train_Loss','Train_Accuracy','Val_Loss','Val_Accuracy','Val_F1'])
                        #<class 'str'> <class 'int'> <class 'list'> <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'numpy.float32'> <class 'torch.Tensor'>
                    
            print('Best Val F1:',str(max_f1),', at epoch:',str(best_f1_epoch))
            entry_append = [subj, i, epoch_lr, train_loss.detach().numpy().tolist(), train_acc.detach().numpy().tolist(), running_loss.detach().numpy().tolist(), 
                            str(label_all_CAT.tolist()), str(predicts.tolist()), acc,  AVG_F1.tolist()]
            print(entry_append)
            subj_result_df.loc[len(subj_result_df)] = entry_append
            
            if (int(max_f1_best_acc) == 1) and (int(max_f1) == 1):
                print('BEST possible scores. Early Stopping for subj',str(subj))
                break
            
            
# FOR EACH SUBJECT, ADD ONCE POST TRAINING #            
            
        subj_result_df.to_csv(os.path.join(model_path, 'subj'+str(subj)+'.csv'), index = None)


        subject_best_f1 = max_f1
    
    
        testing_metrics_entry = [subj, i, best_f1_epoch, val_dataset.__len__(), str(max_pos_label.tolist()), str(max_pos_pred.tolist()), 
                                 str(max_TP.tolist()), str(best_FP.tolist()), str(best_FN.tolist()), max_f1_best_acc, subject_best_f1]
        
        testing_metrics_df.loc[len(testing_metrics_df)] = testing_metrics_entry
        

    #end of all subjects
    testing_metrics_df.to_csv(os.path.join(model_path, 'testing_metrics_df_'+str(args.part)+'.csv') ,index = None)






if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()
    args= parse_args()
    print('EPOCHS', args.epochs)
    print('PART', args.part)
    print('---------')
    print(args)
    print('start training')
    run_training(args)
