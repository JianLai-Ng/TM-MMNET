import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

__DEBUG__ = False

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1, groups=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

def stem3d(in_size, mid_size, out_size): #ADJUSTED
    if __DEBUG__:
                print('start stem conv')
    seq = nn.Sequential(

        nn.Conv3d(in_size, mid_size, kernel_size=(1,3,3), stride=1,padding=(0,1,1),
                            bias=False,groups=1),

        nn.BatchNorm3d(mid_size),
        nn.ReLU(inplace=True),
        nn.Conv3d(mid_size, out_size, kernel_size=(3,1,1), stride=1,padding=(1,0,0),
                    bias=False,groups=1),

        nn.BatchNorm3d(out_size),
        nn.ReLU(inplace=True)
        )
    return seq

    
##CA BLOCK
class CABlock3D (nn.Module): #ADJUSTED
    expansion = 1

    def __init__(self, inplanes, planes, spatial_stride=1, temporal_stride = 1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spatial_kernel = 3, temporal_kernel = 3, temporal_padding = 1, spatial_padding = 1):
        super(CABlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        #################################################################################################
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1,3,3), stride=stride,
        #              padding=dilation, groups=groups, bias=False, dilation=dilation)

        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        print(inplanes, midplanes, planes)
        self.conv1 = nn.Sequential( 
                        nn.Conv3d(
                            inplanes,
                            midplanes,
                            kernel_size=(1, spatial_kernel, spatial_kernel),
                            stride=(1, spatial_stride, spatial_stride),
                            padding=(0, spatial_padding, spatial_padding),
                            bias=False,
                        ),
                    nn.BatchNorm3d(midplanes),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(midplanes, planes, kernel_size=(temporal_kernel, 1, 1), stride=(temporal_stride, 1, 1), padding=(temporal_padding, 0, 0), bias=False)
                    )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        #################################################################################################


        #self.conv2 = conv1x1(planes, planes,groups=groups)
        
        self.conv2 = nn.Conv3d(
                            planes,
                            planes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False,
                        )

        self.bn2 = norm_layer(planes)
        self.attn = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=1, stride=1,bias=False),  # 32*33*33
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.downsample = downsample
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.planes=planes

    def forward(self, x):
        x, attn_last,if_attn =x ##attn_last: downsampled attention maps from last layer as a prior knowledge
        identity = x


        out = self.conv1(x)
        if __DEBUG__:
            print('Within CA BLOCK')
            print('post conv1 sequential: ', out.shape) 
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if __DEBUG__:
            print('identity pre ds: ', identity.shape)   
            print('out: ', out.shape)        
        if self.downsample is not None:
            identity = self.downsample(identity)
            if __DEBUG__:
                print('downsampled')
        if __DEBUG__:
            print('identity post ds: ', identity.shape)   
        out = self.relu(out+identity)
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        attn = torch.cat((avg_out, max_out), dim=1)
        attn = self.attn(attn)
        if __DEBUG__:
            print('attn: ', attn.shape)
        if attn_last is not None:
            if __DEBUG__:
                print('attn_last: ', attn_last.shape)

            attn = attn_last * attn

        attn = attn.repeat(1, self.planes, 1, 1, 1)
        if if_attn:
            out = out *attn
        if __DEBUG__:

            print("")
        return out,attn[:, 0, :, :,:].unsqueeze(1),True





class ResNetTPO(nn.Module): #ADJUSTED   
    print('Creating ResNet2P1 object')
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=4, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 256
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.stem = stem3d(180,210,256)
        

        # self.maxpool = nn.Sequential(
        #     nn.MaxPool3d((1,3,3), stride=(1,2,2), padding =(0,1,1)),
        #     nn.MaxPool3d((3,1,1), stride=(2,1,1), padding =(1,0,0)))

        

        # block, planes, blocks, spatial_stride=1, temporal_stride = 1,spatial_kernel = 3, temporal_kernel = 3, dilate=False,groups=1
        self.layer1 = self._make_layer(block, 256, layers[0], spatial_stride=1, temporal_stride = 1,spatial_kernel = 3, temporal_kernel = 3,
                                       groups=1)
        self.inplanes = int(self.inplanes*1)
        self.maxpool_layer1 = self.make_maxpool_layer(spatial_stride=2, temporal_stride = 2,spatial_kernel = 3, temporal_kernel = 3, temporal_padding = 1)




        self.layer2 = self._make_layer(block, 256, layers[1], spatial_stride=2, temporal_stride = 2,spatial_kernel = 3, temporal_kernel = 3,
                                       dilate=replace_stride_with_dilation[0],groups=1)
        self.inplanes = int(self.inplanes * 1)
        self.maxpool_layer2 = self.make_maxpool_layer(spatial_stride=2, temporal_stride = 1,spatial_kernel = 3, temporal_kernel = 1, temporal_padding = 0)



        self.layer3 = self._make_layer(block, 512, layers[2], spatial_stride=2, temporal_stride = 1,spatial_kernel = 3, temporal_kernel = 2, temporal_padding = 0,
                                       dilate=replace_stride_with_dilation[1],groups=1)
        self.inplanes = int(self.inplanes * 1)
        self.maxpool_layer3 = self.make_maxpool_layer(spatial_stride=2, temporal_stride = 2,spatial_kernel = 3, temporal_kernel = 2)




        self.layer4 = self._make_layer(block, 1024, layers[3],  spatial_stride=2, temporal_stride = 2,spatial_kernel = 3, temporal_kernel = 2, temporal_padding = 0,
                                       dilate=replace_stride_with_dilation[2],groups=1)
        self.inplanes = int(self.inplanes * 1)
        self.maxpool_layer4 = self.make_maxpool_layer(spatial_stride=2, temporal_stride = 2,spatial_kernel = 3, temporal_kernel = 2)




        self.fc = nn.Linear(1024* block.expansion*196, 5)
        self.drop = nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def make_maxpool_layer(self, spatial_stride=1, temporal_stride = 1,spatial_kernel = 3, temporal_kernel = 3, spatial_padding =1, temporal_padding =0):
        mplayer = nn.Sequential(
            nn.MaxPool3d((1,spatial_kernel,spatial_kernel), stride=(1,spatial_stride,spatial_stride), padding =(0,spatial_padding,spatial_padding)),
            nn.MaxPool3d((temporal_kernel,1,1), stride=(temporal_stride,1,1), padding =(temporal_padding,0,0)))
        return mplayer

    def _make_layer(self, block, planes, blocks, spatial_stride=1, temporal_stride = 1,spatial_kernel = 3, temporal_kernel = 3, dilate=False, groups=1, temporal_padding = 1, spatial_padding = 1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= spatial_stride
            spatial_stride = 1
        if spatial_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size = (1,1,1), stride= (1,spatial_stride,spatial_stride)),
                norm_layer(planes * block.expansion),
                nn.Conv3d(planes * block.expansion, planes * block.expansion, kernel_size = (1,1,1), stride= (temporal_stride,1,1)),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, spatial_stride, temporal_stride, downsample=downsample, groups = groups,
                            base_width = self.base_width, dilation = previous_dilation, norm_layer = norm_layer,
                            spatial_kernel = spatial_kernel, temporal_kernel = temporal_kernel, temporal_padding = temporal_padding, spatial_padding= spatial_padding )) #only 1 CA per block
        self.inplanes = planes * block.expansion


        # for _ in range(1, blocks):
        #     layers.append(block(self.inplanes, planes, groups=self.groups,
        #                         base_width=self.base_width, dilation=self.dilation,
        #                         norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x,POS, embedding_au_size):##x->input of main branch; POS->position embeddings generated by sub branch

        if __DEBUG__:
            print('Pre-Stem')
            print('x: ',x.shape)
            print('')

        x = self.stem(x)

        if __DEBUG__:
            print('Post-Stem')
            print('x: ',x.shape)
            print('')
        

        ##### main branch #####

        #L1
        if __DEBUG__:
            print('=================L1=====================')   
        x,attn1,_ = self.layer1((x,None,True))
        if __DEBUG__:
            print('Post Layer 1')
            print('x:', x.shape)
            print('')
            print('attn1:', attn1.shape)
        temp = attn1
        attn1 = self.maxpool_layer1(attn1)

        if __DEBUG__:
            print('pooled attn1:', attn1.shape)
            print('======================================')   
            print('')       
        # Post Layer 1
        # x: torch.Size([16, 256, 6, 112, 112])
        # attn1: torch.Size([16, 1, 6, 112, 112])
        # pooled attn1: torch.Size([16, 1, 6, 56, 56])

        #L2
        if __DEBUG__:
            print('=================L2=====================')   
        x ,attn2,_= self.layer2((x,attn1,True))
        if __DEBUG__:
            print('Post Layer 2')
            print('x:', x.shape)
            print('')
            print('attn2:', attn2.shape)

        attn2=self.maxpool_layer2(attn2)
        if __DEBUG__:
            print('pooled attn1:', attn2.shape)
            print('======================================')   
            print('')   

        #L3
        if __DEBUG__:
            print('=================L3=====================')   
        x ,attn3,_= self.layer3((x,attn2,True))
        if __DEBUG__:
            print('Post Layer 3')
            print('x:', x.shape)
            print('')
            print('attn3:', attn3.shape)
        attn3 = self.maxpool_layer3(attn3)
        if __DEBUG__:
            print('pooled attn1:', attn3.shape)
            print('======================================')   
            print('')   
        
        #L4
        if __DEBUG__:
            print('=================L4=====================')   
        x,attn4,_ = self.layer4((x,attn3,True))
        if __DEBUG__:
            print('Post Layer 4')
            print('x:', x.shape)
            # print('')
            # print('attn4:', attn4.shape)
        if __DEBUG__:
            print('======================================')

        # Post Layer 4
        # x: torch.Size([16, 1024, 1, 14, 14])
        # attn4: torch.Size([16, 1, 1, 14, 14])


        x = torch.squeeze(x, 2)
        # x: torch.Size([batchsize, 1024, 14, 14])
        if __DEBUG__:
            print()
            print()
            print('POST squeeze x:', x.shape)
            print('POS:',POS.shape)


        ##POS torch.Size([batchsize, 196, 1024])
        x=x+POS#fusion of motion pattern feature and position embeddings 
        if __DEBUG__:
            print('x + POS:', x.shape)

        x = torch.flatten(x, 1)
        if __DEBUG__:
            print('POST flattened (x + POS):', x.shape)

        if embedding_au_size != None:
            auc_pred = x[:,:embedding_au_size]
            if __DEBUG__:
                print('embedding_au_size from POST fc flattened (x + POS):', auc_pred.shape)
        else:
            auc_pred = None

        x = self.fc(x) 
        if __DEBUG__:
            print('POST fc flattened (x + POS):', x.shape)



        return x,temp.view(x.size(0),-1), auc_pred

    def forward(self, x,POS, embedding_au_size):
        return self._forward_impl(x,POS, embedding_au_size)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetTPO(block, layers)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model

##main branch consisting of CA blocks
def resnet18_pos_attention(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', CABlock3D, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)



