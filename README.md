# [Background Matters: A Cross-view Bidirectional Modeling Framework for Semi-supervised Medical Image Segmentation (TIP2025)](https://arxiv.org/abs/2505.16625)
by Luyang, Cao, Jianwei Li, and Yinghuan Shi.

### 1. Dependencies
* Python3
* PyTorch>=1.0
* OpenCV-Python, TensorboardX
* NVIDIA GPU+CUDA

### 2. Network Architecture
![figure_arch](https://github.com/caoluyang0830/MSDI/blob/main/fig//framework.png)


### 3. Training and Testing
#### 3.1 Training Process
```
python LA_train.py 
python Pancreas_train.py 
python ACDC_train.py 
```
#### 3.2 Testing Process
```
python test_LA.py
python test_Pancreas.py
python test_ACDC.py
```
### 4. Visual comparison on LA, Pancreas, ACDC, and HRF datasets
![figure_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig//LOLv1.png)
![figure_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig//LOLv2real.png)
![figure_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig//LOLv2sys.png)

### 5. Quantitative comparison on LA, Pancreas, ACDC, and HRF datasets
![table_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig/LOL_v1_result.png)
![table_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig/LOLv2real_result.png)
![table_eval](https://github.com/caoluyang0830/MSDI/blob/main/fig/LOLv2sys_result.png)

## Acknowledgements
Our code is largely based on [BCP](https://github.com/DeepMed-Lab-ECNU/BCP). We sincerely appreciate the valuable contributions of these authors, and we hope our work may also offer some useful insights to the field.

