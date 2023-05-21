# MER 

- Seed
  - random.seed(0)
  - torch.manual_seed(0)
  - np.random.seed(0)


## **GOAL MACRO F1 0.8676**
MER with casme2
with MACRO F1 suggested in https://facial-micro-expressiongc.github.io/MEGC2019/images/MEGC2019%20Recognition%20Challenge.pdf 
_________________________________________

# Architecture Variants
V1-V3 args.part uses range from input string eg. '1-4' processed to train subjects 1,2,3,4

V4 onwards args.part 'hardest' uses subject list '[5,16,7,3,6,17,20,13,26,1]' as default to estimate architecture training effectiveness
_________________________________________


## V1 (MACRO F1 0.8018223965930098):
  - 5 frames (onset, onsetMIDapex, apex, apexMIDoffset, offset) picked out, and 4 difference frames were used
  - 4 channels of input: Vertical and Horizontal Optical Flow maps (original and normalized values), no Action Unit loss implemented.
      normalised eg.vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
  - 70 epochs, lr = 0.0008, gamma = 0.987, plateau at 50 epochs 
  - External 2+1D stem: (4, 128, 180)
  - Internal 2+1D stem: (180,210,256)
  - Attention Module: 256, 256, 512, 1024; POS out-embedding:1024
        
        
## V2 (MACRO F1 0.7817810484941426):
  - 🌻Motion Map: magnititude and angle > HSV > RGB 
  - 🌻3 channels of input
  - 70 epochs, lr = 0.0008, gamma = 0.987, plateau at 50 epochs 
  - 🌻External 2+1D stem: (3, 128, 180)
  - Internal 2+1D stem: (180,210,256)
  - Attention Module: 256, 256, 512, 1024; POS out-embedding:1024
  
  
  
## V3CustomChannel
  - Same channels of input as V1 as it has been observed to increase macro F1 score and decrease epochs required to reach max macro F1 score for respective subjects.
    - 5 frames (onset, onsetMIDapex, apex, apexMIDoffset, offset) picked out, and 4 difference frames were used
    - 4 channels of input: Vertical and Horizontal Optical Flow maps (original and normalized values), no Action Unit loss implemented.
      normalised eg.vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
  - 🌻100 epochs, lr = 0.0008, gamma = 0.987, plateau at 50 epochs 
  - 🌻2+1D convolution prior to connecting to MMnet modules
  - 🌻External 2+1D stem: (4, 180, 256) #kernel size 4 for temporal convolution
  - 🌻Internal 2D stem: (256,256)
  - Attention Module: 256, 256, 512, 1024; POS out-embedding:1024


### V3N (MACRO F1 ): 
  - 🌱**Only normalised**  horizontal & vertical motion map (2 channels)
### V3R (MACRO F1 ): 
  - 🌱**Only raw**  horizontal & vertical motion map (2 channels)
### V3S (MACRO F1 ): 
  - 🌱**Only scaled**  horizontal & vertical motion map (2 channels)
### V3NR (MACRO F1 0.8649977734683617): 
  - 🌱Use **normalised** with **raw** scaled horizontal & vertical motion (4 channels)
  ![image](https://user-images.githubusercontent.com/79305928/236609161-0fa7012d-1eb5-4d82-939f-b2e25cd0de69.png)
### V3RS (MACRO F1 0.822650): 
  - 🌱Use **raw** with **standard** scaled horizontal & vertical motion maps (4 channels)
### V3NS (MACRO F1 0.827281): 
  - 🌱Use **normalised** with **standard** scaled horizontal & vertical motion maps (4 channels)

### V3NRS (MACRO F1 0.826208): 
  - 🌱Use **normalised, raw** with **standard** scaled horizontal & vertical motion (6 channels) 

## V4 (Hard Subjects MACRO F1 < V3):
_downscale architecture from V3 using **External 2+1D stem and onwards**_
  - Same channels of input as V1 as it has been observed to increase macro F1 score and decrease epochs required to reach max macro F1 score for respective subjects.
    - 5 frames (onset, onsetMIDapex, apex, apexMIDoffset, offset) picked out, and 4 difference frames were used
    - 4 channels of input: Vertical and Horizontal Optical Flow maps (original and normalized values), no Action Unit loss implemented.
      normalised eg.vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
  - 2+1D convolution prior to connecting to MMnet modules as per V3
  - External 2+1D stem: (4, 180, 256) #kernel size 4 for temporal convolution
  - 🌻Internal 2D stem: (256,128)
  - 🌻Attention Module: 128, 128, 256, 512; POS out-embedding:512
  - 🌱Training codes and logs
    - 🌱Allow Hardest subject training based on args.part *'hardest'* for all 10 or *'hardest_7'* for hardest 7
    - 🌱Refined logs printed and saved
  - *Hard subjects:*
    - *05, F1 0.507368421052632*
    - *16, F1 0.6*
    - *07, F1 0.666666666666667*
    - *03, F1 0.75*
    - *06, F1 0.666666666666667*
    - *17, F1 0.767655502392345*
    - *20, F1 0.694444444444444*


## V5CustomChannel
_upscale architecture from V3 using **Internal 2D stem**_
  - Same channels of input as V1 as it has been observed to increase macro F1 score and decrease epochs required to reach max macro F1 score for respective subjects.
    - 5 frames (onset, onsetMIDapex, apex, apexMIDoffset, offset) picked out, and 4 difference frames were used
    - 2/4/6 Frames , no Action Unit loss implemented.
      normalised eg.vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
  - 2+1D convolution prior to connecting to MMnet modules as per V3
  - 🌱External 2+1D stem: (4, 180, 512) #kernel size 4 for temporal convolution
  - 🌱Internal 2D stem: (512,512)
  - 🌱Attention Module: 512, 512, 1024, 2048; POS out-embedding:2048

### V5N (MACRO F1 ): 
  - 🌱**Only normalised**  horizontal & vertical motion map (2 channels)
### V5R (MACRO F1 ): 
  - 🌱**Only raw**  horizontal & vertical motion map (2 channels)
### V5S (MACRO F1 ): 
  - 🌱**Only scaled**  horizontal & vertical motion map (2 channels)
### V5NR aka V5 (MACRO F1 0.828984): 
  - 🌱Use **normalised** with **raw** scaled horizontal & vertical motion (4 channels)
    - out[..., 0] = horz_normalized
    - out[..., 1] = horz
    - out[..., 2] = vert
    - out[..., 3] = horz_normalized
  ![image](https://github.com/JianLai-Ng/MER/assets/79305928/f93900a3-84d0-4fac-a718-6f89025e14dc)

### V5RS (MACRO F1 ): 
  - 🌱Use **raw** with **standard** scaled horizontal & vertical motion maps (4 channels)
### V5NS aka V5b (MACRO F1 0.817969): 
  - 🌱Use **normalised** with **standard** scaled horizontal & vertical motion maps (4 channels)
  ![image](https://github.com/JianLai-Ng/MER/assets/79305928/7d69731e-36c8-478a-a5dc-fa008364620f)


### V5NRS (MACRO F1 ): 
  - 🌱Use **normalised, raw** with **standard** scaled horizontal & vertical motion (6 channels) 
  
## V6 (MACRO F1 ):
_upscale architecture from V3 using **Internal 2D stem**_
  - Same channels of input as V1 as it has been observed to increase macro F1 score and decrease epochs required to reach max macro F1 score for respective subjects.
    - 5 frames (onset, onsetMIDapex, apex, apexMIDoffset, offset) picked out, and 4 difference frames were used
    - 4 channels of input: Vertical and Horizontal Optical Flow maps (original and normalized values), Action Unit loss implemented.
      normalised eg.vert_normalized = (vert - vert_min)/(vert_max - vert_min)*(new_max - new_min) + new_min
  - 2+1D convolution prior to connecting to MMnet modules as per V3
  - External 2+1D stem: (4, 180, 512) #kernel size 4 for temporal convolution
  - Internal 2D stem: (512,512)
  - Attention Module: 512, 512, 1024, 2048; POS out-embedding:2048
  - **TRAINING**
    - 🌻Stage 1: Embedding training loop:
        Train and validate based on activation unit embedding loss. **Save best embedding model.**
      - 🌻Add embedding layer for activation unit as penultimate layer
        - _self.embedau = nn.Linear(2048* block.expansion*196, au_size)_
    - 🌻Stage 2: MER training loop:
        **Load best embedding model.** Freeze all pre-trained layers, train final layer and validate based on class loss.
          Train for N epochs, then another M epochs with all layers unfrozen.
        **Save best class model.**
      - 🌻Concat embedding layer to pre-embedding layer as input for final FC layer
        - _self.fc = nn.Linear((2048* block.expansion*196) + au_size, 5)_
    - 🌻New Hyperparameters
      - epochs_S1: 70
      - epochs_S2: 80
      - plateau_epoch1: 50 (51)
      - plateau_epoch2: 50 (51)
      - s2_epoch_unfreeze: 50 (51)
  - *Hard subjects:*
    - *05, F1 0.520652 < 0.563636* -
    - *16, F1 0.600000 = 0.600000* = 
    - *07, F1 0.677778 > 0.649351* +
    - *03, F1 0.750000 > 0.657143* +
    - *06, F1 1.000000 > 0.677778* +

  
## V6c (MACRO F1 ): 
  - 🌱Embeddings to Boolean in train part 2 

_______________________________________
# FUTURE Works

## V5?d (MACRO F1 ): 
  - 🌱Focal loss replacing cross-entropy as loss
    - https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html 
    
## V6?d (MACRO F1 ): 
  - 🌱Focal loss replacing cross-entropy as loss
    - https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html   
_______________________________________
# Best Version justification:
  - V6? = V6 vs V6c, compare F1 of subjects 1-10 as proxy
  - V56? = V6? vs V5 compare, compare F1 of subjects 1-10 as proxy
  - Best V5 = (V56? if V5) vs V5b(only standardscaler) compare F1 of all subjects
_______________________________________
# Ablation study


  - Variations
    - **Original Architecture, New Data Augmentation**
      - with OF vert & hori (4 channels- best variation determined by **V5 vs V5b**) = ?
      - between onset and apex frames (1 frame) as original architecture only takes in 1 frame
      - with customized augmentation techniques (pixel-interpolation & eye-jaw masking)
    - **New Architecture (?), Original Data Augmentation**
      - with original (3 channels) RGB difference
      - between onset, apex and offset frames (4 frames) as new architecture takes in 4 frames
      - with original augmentation techniques
________________________________________

🌻 - represent changes across subsequent versions

🌱 - represent changes the persists
