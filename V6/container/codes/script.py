
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

fs = s3fs.S3FileSystem()

print('v6c')
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
    parser.add_argument('--plateau_epoch1', type=float, default=51,help='ith epoch to start plateauing from stage starting epoch') #51
    parser.add_argument('--plateau_epoch2', type=float, default=51,help='ith epoch to start plateauing from stage starting epoch')  #51
    parser.add_argument('--gamma1', type=float, default=0.987)
    parser.add_argument('--gamma2', type=float, default=0.987)
    parser.add_argument('--part', type=str, default='26')
    parser.add_argument('--data_subdir', type=str, default= training_path, help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=36, help='Batch size.') #as per biggest loso subject 
    parser.add_argument('--lr1', type=float, default=0.0008, help='Initial learning rate for sgd.')
    parser.add_argument('--lr2', type=float, default=0.0008, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs_S1', type=int, default=80, help='Total training epochs for embedding training.') #100
    parser.add_argument('--epochs_S2', type=int, default=70, help='Total training epochs for full model training.') #100
    parser.add_argument('--s2_epoch_unfreeze', type=int, default=35, help='After s2_epoch_unfreeze') #100
    parser.add_argument('--model_dir', type=str, default='s3://{}/{}'.format('2022-mer-s3bucket', 'models/'))


    return parser.parse_args()

# For iterable-style datasets, since each worker process gets a replica of the dataset object, naive multi-process loading will often result in duplicated data. Using torch.utils.data.get_worker_info() and/or worker_init_fn, users may configure each replica independently. (See IterableDataset documentations for how to achieve this. ) For similar reasons, in multi-process loading, the drop_last argument drops the last non-full batch of each worker’s iterable-style dataset replica.


class RafDataSet(data.Dataset):
    def __init__(self, data_subdir, phase,num_loso, basic_aug = False):
        self.phase = phase
        self.transform = None
        self.data_subdir = data_subdir +'/'
        self.transform_norm = None
        self.i_crop = 0
        self.j_crop = 0
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
 
                label_au = label_au.split("+")
                self.label_au.append([label_au])


        flatten_list = list(chain.from_iterable(self.label_au))
        self.au_set= list(set([str(a) for a in flatten_list]))
        self.au_set.sort()
        # self.embedding_au_size = len(self.au_set)
        # self.au_set_to_index = {}
        # index = 0
        # for an_ele in self.au_set:
        #     self.au_set_to_index[an_ele] = index 
        #     index+=1
        



            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]
    def compute_optflow_magsincos(self, prev_image, current_image):

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

        new_min, new_max = -1, 1

        #HORZ
        horz_min, horz_max = horz.min(), horz.max()
        horz_normalized = (horz - horz_min)/(horz_max - horz_min)*(new_max - new_min) + new_min
        #horz_interpolated = np.interp(horz, (horz.min(), horz.max()), (-1, +1))

        #VERT
        vert_min, vert_max = vert.min(), vert.max()
        vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
        vert_interpolated = np.interp(vert, (vert.min(), vert.max()), (-1, +1))



        out = np.zeros((224,224,4), dtype = float)
        out[..., 0] = horz_normalized
        out[..., 1] = horz
        out[..., 2] = vert
        out[..., 3] = vert_normalized
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

   



        #get optical flow map
        #print(type(np.asarray(image_on0)))
        of1 = self.compute_optflow_magsincos(np.asarray(image_on0), np.asarray(image_onmid0))
        of2 = self.compute_optflow_magsincos(np.asarray(image_onmid0), np.asarray(image_mid0))
        of3 = self.compute_optflow_magsincos(np.asarray(image_mid0), np.asarray(image_midoff0))
        of4 = self.compute_optflow_magsincos(np.asarray(image_midoff0), np.asarray(image_off0))
       


        return transforms.ToTensor()(image_on0),transforms.ToTensor()(of1), transforms.ToTensor()(of2), transforms.ToTensor()(of3), transforms.ToTensor()(of4),label_all, str(label_au)

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




class MMNetEmbed(nn.Module): #ADJUSTED
    def __init__(self, embedding_au_size = 35):
        super(MMNetEmbed, self).__init__()

        self.embedding_au_size = embedding_au_size

        self.conv_act = nn.Sequential(

            nn.Conv3d(in_channels=4, out_channels=180, kernel_size=(1,3,3), stride=(1,2,2),padding=(0,1,1), bias=False,groups=1),
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
        self.main_branch =resnet18_pos_attention(au_size = embedding_au_size)

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
        embedding =self.main_branch(act,POS, self.embedding_au_size)

        return embedding



class MMNet(nn.Module): #ADJUSTED
    def __init__(self, pretrained_model, embedding_au_size = None, ):
        super(MMNet, self).__init__()
    
        self.embedding_au_size = embedding_au_size
        self.pretrained = pretrained_model
        self.last_layer = nn.Linear((2048* 1*196) + embedding_au_size, 5)

    def forward(self, onset, x1, x2, x3, x4, if_shuffle):


        pre_embeddings, embeddings = self.pretrained(onset, x1, x2, x3, x4, if_shuffle)
        bool_embeddings = (torch.sigmoid(embeddings)>0.5).float()
        x = self.last_layer(torch.cat((pre_embeddings, bool_embeddings), 1))

        return x
    def freeze_all_but_last(self):
    
        #named_parameters is a tuple with (parameter name: string, parameters: tensor)
        for n, p in self.named_parameters():
            if 'last' in n:
                pass
            else:
                p.requires_grad = False


def run_training(args):

    criterion1 = torch.nn.BCEWithLogitsLoss()
    criterion2 = torch.nn.CrossEntropyLoss()

    if 'hardest' in args.part :
        if '_' not in args.part:
            LOSO = [5,16,7,3,6,17,20,13,26,1]
        else:
            top_n = int(args.part.split('_')[-1])
            LOSO = [5,16,7,3,6,17,20,13,26,1][:top_n]
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
    print('Epochs:',args.epochs_S1 + args.epochs_S2)
    
    #[subj, i, best_f1_epoch, val_dataset.__len__(), str(max_pos_label.tolist()), str(max_pos_pred.tolist()), str(max_TP.tolist()), subject_best_f1]
    embed_testing_metrics_df = pd.DataFrame(columns = ['subject','total_trained_epochs', 'best_epoch', 'val_dataset_len', 
                                                       'truth' ,'prediction', 'best_macro_f1'])

    testing_metrics_df = pd.DataFrame(columns = ['subject','total_trained_epochs', 'best_epoch', 'val_dataset_len', 
                                                 'truth' ,'prediction', 'best_accuracy_under_best_macro_f1', 'best_macro_f1'])





    for subj in LOSO:
        subj_result_df = pd.DataFrame(columns = ['Subject', 'Epoch','LR', 'STAGE','Train_Loss',
                                                 'Train_F1','Val_Loss', 
                                                 'truth' ,'prediction','Val_Acc','Val_F1'])

        train_dataset = RafDataSet(args.data_subdir, phase='train', num_loso=subj,
                                   basic_aug=True)
        val_dataset = RafDataSet(args.data_subdir, phase='test', num_loso=subj)
        
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
        

        #AU to embedding dictionary
        ausets = [eval(a) for a in train_dataset.au_set + val_dataset.au_set]
        flatten_list = list(chain.from_iterable(ausets))
        ausets= list(set([str(a) for a in flatten_list]))
        ausets.sort()
        

        index = 0
        au_set_to_index = {}
        for an_ele in ausets:
            au_set_to_index[an_ele] = index 
            index+=1
        print('au_set_to_index: '+ str(au_set_to_index))



 
      
#### EMBEDDING TRAINING

        ##model initialization
        embed_net = MMNetEmbed(embedding_au_size = len(ausets))
        embed_net_params = embed_net.parameters()
        optimizer_1 = torch.optim.AdamW(embed_net_params, lr=args.lr1, weight_decay=0.6)
        scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, gamma=args.gamma1)

        max_f1_min_loss = 99999
        max_f1 = 0

        for i in range(1, int(args.epochs_S1)+1):
            epoch_lr1 = scheduler_1.get_last_lr()[0]
            running_loss1 = 0.0

            predict_appends = None
            truth_appends = None

            iter_cnt1 = 0

            embed_net.train()


            for batch_i, (onset, of1, of2, of3,  of4, label_all,
            label_au) in enumerate(train_loader):
                batch_sz = of1.size(0)
                b, c, h, w = of1.shape
                iter_cnt1 += 1
                
                
                onset = onset#.cuda()
                of1 = of1#.cuda()
                of2 = of2#.cuda()
                of3 = of3#.cuda()
                of4 = of4#.cuda()

                label_all = label_all#.cuda()
                label_au = label_au#.cuda()

                ##train MMNet
                pre_emb, emb_au = embed_net(onset, of1, of2, of3, of4, False)

                #### au_embeddings process embeddings from dataset-loader string version to embeddings torch 
                #MAKE au_embeddings (truth)
                batch_aus_raw = [eval(a)[0] for a in label_au]

                au_embeddings = torch.zeros(batch_sz,len(ausets))
                sample_n = 0
                for an_instance in batch_aus_raw:
                    for an_au in an_instance:
                        au_embeddings[sample_n][au_set_to_index[an_au]] = 1
                    sample_n+=1
                #au_embeddings = au_embeddings.detach().numpy()
                #######################################    
    
                
                #backward prop
                loss_1 = criterion1(emb_au, au_embeddings) #BCElogit has built in sigmoids
                optimizer_1.zero_grad()
                loss_1.backward()
                optimizer_1.step()


                #### process embeddings from model output- apply sigmoid and threshold
                #MAKE predicts1 (prediction)
                predicts1 = (torch.sigmoid(emb_au)>0.5).float()

                # Iteration F1
                S1_f1_train_iter = f1_score(au_embeddings, predicts1, average='macro', zero_division=0)
 


                # print('        S1 iter:' +str(iter_cnt))
                # print('        S1 time:' + datetime.now().strftime("%d-%m-%Y, %H:%M:%S"))
                # print('        S1 loss:' +str(loss_1))
                # print('        S1 F1:' +str(S1_f1_train))

                print(        'S1 iter:%s, S1 time:%s, S1 loss:%s, S1 F1:%s'%( str(iter_cnt1), datetime.now().strftime("%d-%m-%Y, %H:%M:%S"), str(loss_1), str(S1_f1_train_iter) ))
                
                #for epoch stats
                running_loss1+=loss_1
                if predict_appends is None:
                    predict_appends = predicts1
                else:
                    predict_appends = torch.cat((predict_appends, predicts1), axis=0)
                if truth_appends is None:
                    truth_appends = au_embeddings
                else:
                    truth_appends= torch.cat((truth_appends, au_embeddings), axis=0)
                
            

            ## lr decay

            if i>=0:
                S1_f1_train_epoch = f1_score(truth_appends, predict_appends, average='macro', zero_division=0)

                print('[Epoch %d] Training F1: %.4f. Loss: %.3f' % (i, S1_f1_train_epoch, running_loss1))

            if i < args.plateau_epoch1:

                scheduler_1.step()
            else:
                print('LR Plateau')

            embed_train_loss = running_loss1
            embed_train_f1 = S1_f1_train_epoch
            

            test_predict_appends = None
            test_truth_appends = None


            with torch.no_grad():
                running_loss = 0.0
                bingo_cnt = 0
                sample_cnt = 0
                pre_lab_all = []
                Y_test_all = []
                embed_net.eval()
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
                    pre_emb, emb_au = embed_net(onset, of1, of2, of3, of4, False)



                    #### au_embeddings process embeddings from dataset-loader string version to embeddings torch 
                    #MAKE au_embeddings (truth)
                    batch_aus_raw = [eval(a)[0] for a in label_au]

                    au_embeddings = torch.zeros(batch_sz,len(ausets))
                    sample_n = 0
                    for an_instance in batch_aus_raw:
                        for an_au in an_instance:
                            au_embeddings[sample_n][au_set_to_index[an_au]] = 1
                        sample_n+=1
                    #au_embeddings = au_embeddings.detach().numpy()
                    
                    #######################################    


                    #### process embeddings from model output- apply sigmoid and threshold
                    #MAKE predicts1 (prediction)
                    predicts1 = (torch.sigmoid(emb_au)>0.5)



                    if test_predict_appends is None:
                        test_predict_appends = predicts1
                    else:
                        test_predict_appends = torch.cat((test_predict_appends, predicts1), axis=0)

                    if test_truth_appends is None:
                        test_truth_appends = au_embeddings
                    else:
                        test_truth_appends= torch.cat((test_truth_appends, au_embeddings), axis=0)
                        
                #test_truth_appends = test_truth_appends.detach().numpy()

                embed_test_loss = criterion1(emb_au, au_embeddings)
                embed_test_f1 = f1_score(test_truth_appends, test_predict_appends, average='macro', zero_division=0)
                print('test_truth_appends & test_predict_appends')
                print(test_truth_appends)
                print(test_predict_appends)


                #get human readable embeddings
                test_labels_embedding_inter = np.nonzero(test_truth_appends.detach().numpy() > 0)
                test_labels_embedding_inter = np.transpose(test_labels_embedding_inter)
                test_label_embeds = {a[0]:[[key for key, val in au_set_to_index.items() if val == b[1]][0] for b in test_labels_embedding_inter if b[0] ==a[0]]for a in test_labels_embedding_inter}

                test_pred_embedding_inter = np.nonzero(test_predict_appends.float().detach().numpy() > 0)
                test_pred_embedding_inter = np.transpose(test_pred_embedding_inter)
                test_pred_embeds = {a[0]:[[key for key, val in au_set_to_index.items() if val == b[1]][0] for b in test_pred_embedding_inter if b[0] ==a[0]]for a in test_pred_embedding_inter}

                if embed_test_f1 == max_f1:
                    if abs(embed_test_loss) < abs(max_f1_min_loss):
                        best_f1_epoch = i
                        max_f1 = embed_test_f1
                        max_f1_min_loss = embed_test_loss
                        max_pos_label = test_label_embeds
                        max_pos_pred = test_pred_embeds

                        print('NEW lowest loss with Best Val F1 for subj'+str(subj) +':',str(max_f1)+', '+str(max_f1_min_loss)+ 'at epoch:'+str(best_f1_epoch))
                        print('test label:', str(max_pos_label))
                        print('pred label:', str(max_pos_pred))
                        #save model
                        
                        with open(os.path.join(model_path, str(subj)+'embed_model.pth'), 'wb') as f:
                            torch.save(embed_net.state_dict(), f)
                        print('MODEL SAVED AS' +  str(subj)+'embed_model.pth')

                elif embed_test_f1 > max_f1:
                    best_f1_epoch = i
                    max_f1 = embed_test_f1
                    max_f1_min_loss = embed_test_loss
                    max_pos_label = test_label_embeds
                    max_pos_pred = test_pred_embeds
                    print('NEW Best Val F1 for subj'+str(subj) +':',str(max_f1)+', at epoch:'+str(best_f1_epoch))
                    print('test label:', str(max_pos_label))
                    print('pred label:', str(max_pos_pred))
                    #save model
                    
                    with open(os.path.join(model_path, str(subj)+'embed_model.pth'), 'wb') as f:
                        torch.save(embed_net.state_dict(), f)
                    print('MODEL SAVED AS' +  str(subj)+'embed_model.pth')

                print("[Epoch %d] Validation F1:%.4f. Loss:%.3f" % (i, embed_test_f1, embed_test_loss ))
            

            print('Best Val F1:',str(max_f1),', at epoch:',str(best_f1_epoch))

            #['Subject', 'Epoch','LR', 'Train_Loss','Train_F1',
                # Val_Loss', 'truth' ,'prediction','Val_F1']
        # subj_result_df = pd.DataFrame(columns = ['Subject', 'Epoch','LR', 'STAGE',
        #                                           'Train_Loss',
        #                                          'Train_F1','
    #                                               Val_Loss', 
        #                                          'truth' ,'prediction',
        #                                           'Val_Acc', 'Val_F1'])
            entry_append = [subj, i, epoch_lr1, 'embeddings', 
                            embed_train_loss.detach().numpy().tolist(), 
                            embed_train_f1, 
                            embed_test_loss.detach().numpy().tolist(), 
                            str(max_pos_label), str(max_pos_pred), 'na', 
                            embed_test_f1]
            print(entry_append)
            subj_result_df.loc[len(subj_result_df)] = entry_append
            
            if (int(max_f1_min_loss) == 0) and (int(max_f1) == 1):
                print('BEST possible scores. Early Stopping for subj',str(subj))
                break



        #subject specific
        subject_best_f1 = max_f1 
        embed_testing_metrics_entry = [subj, i, best_f1_epoch, val_dataset.__len__(), 
                                       str(max_pos_label), str(max_pos_pred)  ,subject_best_f1]
        embed_testing_metrics_df.loc[len(embed_testing_metrics_df)] = embed_testing_metrics_entry
        
############################################################################################################################  
        #LOAD BEST EMBEDDING MODEL
        embed_net = MMNetEmbed(embedding_au_size = len(ausets))
        embed_net.load_state_dict(torch.load( os.path.join(model_path, str(subj)+'embed_model.pth' )))

############################################################################################################################        
        mmnet = MMNet(embed_net, len(ausets))
        mmnet_params = mmnet.parameters()
        optimizer_2 = torch.optim.AdamW(mmnet_params, lr=args.lr2, weight_decay=0.6)
        scheduler_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer_2, gamma=args.gamma2)   

        
        max_f1_best_acc = 0
        max_f1 = 0
        

        for i in range(int(args.epochs_S1)+1, int(args.epochs_S1) + int(args.epochs_S2)+1):
            ## lr decay
            if i <= args.epochs_S1 + args.s2_epoch_unfreeze:
                mmnet.freeze_all_but_last() #freeze up to embeddings layer
                print('pretrained layers frozen')
           
            else:
                mmnet.train()
                print('train all layers')

            epoch_lr = scheduler_2.get_last_lr()[0]
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0


            predict_appends = None
            truth_appends = None

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
                output_class_premax = mmnet(onset, of1, of2, of3, of4, False)

                loss_all = criterion2(output_class_premax, label_all)

                optimizer_2.zero_grad()

                loss_all.backward()

                optimizer_2.step()
                running_loss += loss_all
                _, predicts = torch.max(output_class_premax, 1)
                correct_num = torch.eq(predicts, label_all).sum()
                correct_sum += correct_num

                if predict_appends is None:
                    predict_appends = predicts
                else:
                    predict_appends = torch.cat((predict_appends, predicts), axis=0)
                    
                if truth_appends is None:
                    truth_appends = label_all
                else:
                    truth_appends= torch.cat((truth_appends, label_all), axis=0)

                # print('        iter:' +str(iter_cnt))
                # #print('        mem allocated {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))
                # print('        time:' + datetime.now().strftime("%d-%m-%Y, %H:%M:%S"))
                # print('        loss:' +str(loss_all))
                print(        'S2 iter:%s, S2 time:%s, S2 loss:%s, S2 F1:%s'%( str(iter_cnt), datetime.now().strftime("%d-%m-%Y, %H:%M:%S"), str(loss_all), str(S1_f1_train_iter) ))

            

            ## lr decay

            if i>=0:
                acc = correct_sum.float() / float(train_dataset.__len__())
                S2_f1_train_epoch = f1_score(truth_appends, predict_appends, average='macro', zero_division=0)
                running_loss = running_loss / iter_cnt

                print('[Epoch %d] Training accuracy: %.4f. Training F1: %.4f. Loss: %.3f' % (i, acc, S2_f1_train_epoch, running_loss))
            if i < args.epochs_S1 + args.plateau_epoch2:

                scheduler_2.step()
            else:
                print('LR Plateau')

            train_loss = running_loss
            train_acc = acc
            

            ALL_CAT = None
            label_all_CAT = None

            with torch.no_grad():
                running_loss = 0.0
                bingo_cnt = 0
                sample_cnt = 0
                pre_lab_all = []
                Y_test_all = []
                mmnet.eval()
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
                    ALL = mmnet(onset, of1, of2, of3, of4, False)

                    if ALL_CAT == None:
                        ALL_CAT = ALL
                    else:
                        ALL_CAT = torch.cat((ALL_CAT, ALL), 0)
                        
                    if label_all_CAT == None:
                        label_all_CAT = label_all
                    else:
                        label_all_CAT = torch.cat((label_all_CAT, label_all), 0)
                        
                loss = criterion2(ALL_CAT, label_all_CAT)
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
                
                AVG_F1 = f1_score(label_all_CAT, predicts, average='macro')
                


                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)  
    
                acc = np.around(acc.detach().numpy(), 4)


                if AVG_F1 == max_f1:
                    if max_f1_best_acc < acc:
                        best_f1_epoch = i
                        max_f1 = AVG_F1
                        max_f1_best_acc = acc
                        max_pos_label = label_all_CAT
                        max_pos_pred = predicts

                        print('NEW ACC wtih Best Val F1 for subj'+str(subj) +':',str(max_f1)+', '+str(acc)+ 'at epoch:'+str(best_f1_epoch))
                        #save model
                        
                        with open(os.path.join(model_path, str(subj)+'model.pth'), 'wb') as f:
                            torch.save(mmnet.state_dict(), f)
                        print('MODEL SAVED AS' +  str(subj)+'model.pth')

                elif AVG_F1 > max_f1:
                    best_f1_epoch = i
                    max_f1 = AVG_F1
                    max_f1_best_acc = acc
                    max_pos_label = label_all_CAT
                    max_pos_pred = predicts

                    print('NEW Best Val F1 for subj'+str(subj) +':',str(max_f1)+', at epoch:'+str(best_f1_epoch))
                    #save model
                    
                    with open(os.path.join(model_path, str(subj)+'model.pth'), 'wb') as f:
                        torch.save(mmnet.state_dict(), f)
                    print('MODEL SAVED AS' +  str(subj)+'model.pth')
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, sklearn_F1-score:%.3f" % (i, acc, running_loss,  AVG_F1))
            

            print('Best Val F1:',str(max_f1),', at epoch:',str(best_f1_epoch))
            entry_append = [subj, i, epoch_lr, 'classes', 
                            train_loss.detach().numpy().tolist(), 
                            S2_f1_train_epoch.tolist(),
                            running_loss.detach().numpy().tolist(), 
                            str(label_all_CAT.tolist()), str(predicts.tolist()), 
                            acc,  AVG_F1.tolist()]
            

            print(entry_append)
            subj_result_df.loc[len(subj_result_df)] = entry_append
            
            if (int(max_f1_best_acc) == 1) and (int(max_f1) == 1):
                print('BEST possible scores. Early Stopping for subj',str(subj))
                break           
# FOR EACH SUBJECT, ADD ONCE POST TRAINING #            



############################################################################################################################        
        subj_result_df.to_csv(os.path.join(model_path, 'subj'+str(subj)+'.csv'), index = None)

        subject_best_f1 = max_f1
    
    
        testing_metrics_entry = [subj, i, best_f1_epoch, val_dataset.__len__(), str(max_pos_label.tolist()), str(max_pos_pred.tolist()),
                                  max_f1_best_acc, subject_best_f1]
        
        testing_metrics_df.loc[len(testing_metrics_df)] = testing_metrics_entry
        

    #end of all subjects
    embed_testing_metrics_df.to_csv(os.path.join(model_path, 'embeddings_testing_metrics_df_'+str(args.part)+'.csv') ,index = None)
    testing_metrics_df.to_csv(os.path.join(model_path, 'testing_metrics_df_'+str(args.part)+'.csv') ,index = None)







if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()
    args= parse_args()
    print('Embedding training: EPOCHS 1 to %s, plateau at %s'%(str(args.epochs_S1) ,str(args.plateau_epoch1) ))
    print('FC training: EPOCHS %s to %s, plateau after %s, unfreeze all after %s'%(str(args.epochs_S1+1), 
                                                                                str(int(args.epochs_S1) + int(args.epochs_S2)) ,
                                                                                str(args.epochs_S1 + args.plateau_epoch2 ),
                                                                                str(args.epochs_S1 + args.s2_epoch_unfreeze)))
    print('PART', args.part)
    print('---------')
    print(args)
    print('start training')
    run_training(args)
