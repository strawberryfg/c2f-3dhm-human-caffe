# c2f-3dhm-human-caffe
This is the caffe reimplementation of "Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose". (\emph{citation})

You can find screenshots of eval on test set in figs/ (**d2 =  16, 32, 64**). (Random or full test)

----
## Overture
- I write C++ faster than Python.
- I write faster C++ than Python.
- I know C++ / Caffe is not easy to understand.


----
## Your briefing
2-stage Hourglass (**d1 = 1, d2 = 16/32/64**) w/ **batch size = 3** 🌝 🌚 😈

Exquisite **ResNet w/ integral** coming up soon.  💪

**Batch size = 1** w/ **group normalization** trained on **windows caffe** coming up soon. 💩

See Guideline.pdf for detailed description.

----
## Environment
- Ubuntu / Windows

For Ubuntu, I used **two 12 GB TITAN Xp**. For Windows, I used one **8 GB GTX 1070** on **Alienware PC**. 

----
## Data
- See "Data" section in Guideline.pdf for joint definition. (full 32 vs usable 16)

| Folder Name | Download Link | Description | A Toy Example |
|:-:|:-:|:-:|:-:|
| bbx_all_new | [bbx](https://drive.google.com/file/d/1TsPBkrkITMK2Snzb0WcUzOSVz3EIjr1p/view?usp=sharing) | (bbx_x1, bbx_y1, bbx_x2, bbx_y2) | [bbx](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/bbx_all_new/9.txt) |
| center_x | [center_x] | center_x (constant: 112.0) | [center_x](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/center_x/18.txt) |
| center_y | [center_y] | center_y (constant: 112.0) | [center_y](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/center_y/13.txt) |
| scale | [scale] | person image scale (constant: 224.0) | [scale](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/scale/23.txt) |
| gt_joint_2d_raw_new | [gt_2d] | 2d gt on 224x224 cropped image (32x2) | [gt_2d](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/gt_joint_2d_raw_new/1000.txt) |
| image_path_file |  | image path for each sample | [img_path_file](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/image_path_file/20.txt) |
| gt_joint_3d_mono_raw | [gt_3d](https://drive.google.com/open?id=1EI8AKCorNqXPSvt3tjfLWW9YX1WsXSIP) | monocular 3d gt in camera coordiante (32x3) | [gt_3d](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/gt_joint_3d_mono_raw/1000006.txt)|
| camera_all | [camera](https://drive.google.com/open?id=1hIjwJdc6bAaaKgDmUuVbLrRDGW3zIKMa) | intrinsic & extrinsic camera parameters | [camera](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/camera_all/100.txt) |
| index_range | [ind_range](https://github.com/strawberryfg/c2f-3dhm-human-caffe/tree/master/data/full/index_range) | index range per (subject, action) | |
| info_all | [basic_info](https://drive.google.com/open?id=1zzK9ysvdMs58gaCGvASLDxj0zXL6qzaM) | video/action name/subaction/camera id/frame id | [basic_info](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/info_all/1000019.txt) |
| images | [img](https://drive.google.com/open?id=1IwqkZBu4tZBTtNbOJc9Jduxpvcvf68xC) | all the cropped images 224x224 | |

## Trained models
| Method |d2   |  MPJPE(mm)  | Caffe Model  | Solver State |
|:-:|:-:|:-:|:-:|:-:|
| Mine     | 64 |      | [Google Drive (net_iter_720929.caffemodel)](https://drive.google.com/file/d/13-rF6drfrEyuiF4u-UtqycbLm6Msc8RP/view?usp=sharing) | [Google Drive (net_iter_720929.solverstate)](https://drive.google.com/file/d/1t7pkS88-8IqUAxIYcbUKROfFTrPnhLfr/view?usp=sharing) |
| Mine     | 32 | [68.6](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/figs/test_d32_full.png) | [Google Drive (net_iter_640000.caffemodel)](https://drive.google.com/file/d/1Q5ztDnossLMKoaZEwTsjdLYvK_BJDTrW/view?usp=sharing) | [Google Drive (net_iter_640000.solverstate)](https://drive.google.com/file/d/1rdxfEOwVngvyBXPSYh_1xPUPlmBOV2hS/view?usp=sharing) |
| Mine     | 16 | [73.6](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/figs/test_d16_full.png) | [Google Drive (net_iter_560000.caffemodel)](https://drive.google.com/open?id=1-sUW2vGWtgeZxUqUznSvlsAQW3e6DiQz) | [Google Drive (net_iter_560000.solverstate)](https://drive.google.com/file/d/1XmzV7HuIdMEkeRX-fY0dfXypvxrFyoGB/view?usp=sharing) |
| C2F[1]     | 64 | 69.8 | None | None |
| Integral[2] | 64 | 68.0 | None | None |

[1] Pavlakos, Georgios, et al. "Coarse-to-fine volumetric prediction for single-image 3D human pose." Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 2017.

[2] Sun, Xiao, et al. "Integral human pose regression." arXiv preprint arXiv:1711.08229 (2017).

Included for reference.

----
## Kick off the testing
As you know, evaluation on the entire dataset takes time. For testing on a random subset, I implemented a random index generation layer. See screenshot **"figs/test_d64_rand.png"**, **"figs/test_d32_rand.png"**, **"figs/test_d16_rand.png"** for details.

I should claim that this is just for fun, please do not not take it seriously. You might get, say, **68.2 mm** and **68.4 mm** in two different runs.

- **d2 =  64**
  ```
  cd testing
  caffe test -model test_d64_rand.prototxt -weights models/net_iter_720929.caffemodel -iterations 500
  ```
  This will give you **figs/rand_test_d64.png** (unstable number around **68 mm** due to small number of samples)
  
- **d2 =  32**
  ```
  caffe test -model test_d32_rand.prototxt -weights models/net_iter_640000.caffemodel -iterations 500
  ```
  This will give you **figs/rand_test_d32.png** (unstable number around **71 mm**)
  
- **d2 =  16**
  ```
  caffe test -model test_d16_rand.prototxt -weights models/net_iter_560000.caffemodel -iterations 500
  ```
  This will give you **figs/rand_test_d16.png** (unstable number around **74 mm**)
  
  
----
## Full testing
For full evaluation on H36M test set
 
- **d2 =  64**
  ```
  cd testing
  caffe test -model test_d16_statsfalse.prototxt -weights models/net_iter_720929.caffemodel -iterations 183000
  ```
  This will give you mm (**figs/test_d64_full.png**)


- **d2 =  32**
  ```
  caffe test -model test_d32_statsfalse.prototxt -weights models/net_iter_640000.caffemodel -iterations 183000
  ```
  This will give you **68.6 mm** (**figs/test_d32_full.png**)
 
  
- **d2 =  16**
  ```
  caffe test -model test_d16_statsfalse.prototxt -weights models/net_iter_560000.caffemodel -iterations 183000
  ```
  This will give you **73.6 mm** (**figs/test_d16_full.png**)

## Training

Training is a bit tricky. For a comprehensive interpretation, see pdf. Here's the thing:

- I started with **d2 =  2** to warm up. Simply run 
  ```
  cd training 
  caffe train --solver=solver_d2.prototxt 
  ```
  I trained from scratch w/o MPII 2D HM pretraining, with **2.5e-5** as base_lr and **RMSProp**. 2 GPUs were used unless otherwise specified. Weight initialization is gaussian w/ **0.01 std**. Loss ratio of 3d HM to 2d HM is **0.1:1**.
  
- **d2 =  4** Finetune weights from **d2 =  2** after convergence.
  ```
  caffe train --solver=solver_d4.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **137 mm** on train and **150 mm** on test. For eval on training set, simply uncomment **"index_lower_bound: 0" "index_upper_bound: 1559571"** of **"GenRandIndex"** layer. Loss ratio is **0.3:1**.
 
- **d2 =  8** Finetune weights from **d2 =  4** after convergence.
  ```
  caffe train --solver=solver_d8.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **72 mm** on train and **86 mm** on test. Loss ratio is **0.1:1**.
- **d2 =  16** Finetune weights from **d2 =  8** after convergence 
  ```
  caffe train --solver=solver_d16.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **47 mm** on train and **72 mm** on test. Loss ratio is **0.03:1**.

- **d2 =  32** Finetune weights from **d2 =  16** after net_iter_560000.solverstate 
  ```
  caffe train --solver=solver_d32.prototxt --snapshot=net_iter_560000.solverstate 
  ```
  You will get around **39 mm** on train and **71 mm** on test. Loss ratio is **0.03:1**.
  I changed the weight initialization of 3D heatmap to normal distribution with **0.001 std** in place previous 0.01 as I found the MPJPE error did not slump.
  
- **d2 =  64** Finetune weights from **d2 =  32** after net_iter_640000.solverstate 
  ```
  caffe train --solver=solver_d64.prototxt --snapshot=net_iter_640000.solverstate 
  ```
  You will get around **37 mm** on train and **68 mm** on test. Loss ratio is **0.03:1**. I again changed weight initialization of 3D heatmap from 0.001 gaussian $\rightarrow$ **0.0003**. 
  
This sounds pretty sketchy, right? Another way to train this is simply train **d1 = 1, d2 = 64** from scratch. Details:
\emph{missing, TO DO}

### Notes:

- I set **use_global_stats** to **false** during inference due to small batch size, otherwise you would get a totally different MPJPE number. I cannot recall the paper that mentioned it. Let me find the paper.

- The major differences between prototxts lies in:

   **a)** depth dimension param (Use sublime or notepad++ to search keywords **"depth_dims"**)
   
   **b)** 3d heatmap slicing layer. (Simply search **"cube_"**)
   
   **c)** 3d heatmap reshaping layer (**"heatmap2_flat_scale"**)
   
   **d)** loss ratio of 3d heatmap and 2d heatmap. Basic rule is magnitude of these two losses should be the same.
   
   **e)** different weight initialization of last conv layer for 3d heatmap.
   
- I only used **L2 loss** during training. Nevertheless I have **smooth L1 loss**, **adaptive loss**, and **integral loss** in prototxt, as can be seen in figs/*.png. Adaptive loss tries to automatically balance weight magnitude of different euclidean regression loss. See pdf for details about integral loss.

- MPJPE error of **argmax** operation is **"error(mm)_3d_s2_max"**.