# HVPNet
HVPNet: A Unified Bio-Inspired Network for General Salient and Camouflaged Object Detection
 1. Model Selection
Choose the appropriate model version based on your hardware capabilities and accuracy requirements:

Standard Version: HVPNet  
Lightweight Version: HVPNet(-)

 2. Task Selection
Select the relevant model version according to your task and performance needs:

Standard Version: HVPNet (SMT-t + MobileNetV2)
Lightweight Version: HVPNet(-) (MobileNetV2 + MobileNetV2)

 3. Dataset Preparation
We employed the following datasets for training the model:

RGB SOD: DUTS (Wang et al., 2017)

RGB-D SOD: NJUD (Ju et al., 2014), NLPR (Peng et al., 2014), DUTLF-Depth (Piao et al., 2019)

RGB-T SOD: VT5000 (Tu et al., 2022)

VSOD: DAVIS, FBMS, DAVSOD

RGB COD: COD10K (Fan et al., 2020), CAMO (Le et al., 2019)

RGB-D COD: COD10K, CAMO

VCOD: MoCA-Mask
├── dataset
│ ├── RGB
│ ├── Depth
│ └── GT

Ensure that each dataset (RGB, Depth, GT) is placed in the corresponding folder.

 4. Pretrained Weights
Download the pretrained weights for your selected model:

SMT-t
MobileNetV2

The pretrained weights can be found in the respective links provided in the documentation.

5. Training
To train the model from scratch, execute the following command:
python train_Net.py
Ensure that your environment is properly set up with the necessary dependencies before training.

6. Testing
For running inference and testing the model, use the following command:

python test_Net.py
This will evaluate the trained model on the test set and generate results.
