# c2f-3dhm-human-caffe
This is the caffe reimplementation of [Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose](https://arxiv.org/pdf/1611.07828.pdf)

You can find screenshots of eval on test set in figs/ (**d2 =  16, 32, 64**). (Random or full test)

----
## News
- Reaches **67.1** mm on entire test set!

----
## Overture
- I write C++ faster than Python.
- I write faster C++ than Python.
- I know C++ / Caffe is not easy to understand.
- People don't like Netscape Browser or iPhone 4s any more. hmmmmmm.
- People tend to use Hourglass for human pose while powerful ResNet is enough. hmmmmmmmmmmmmmm.


----
## Your briefing
2-stage Hourglass (**d1 = 1, d2 = 16/32/64**) w/ **batch size = 3** 🌝 🌚 😈

Exquisite **ResNet w/ integral** coming up soon.  💪

**Batch size = 1** w/ **group normalization** trained on **windows caffe** coming up soon. 💩

Caffe Hourglass is imported from [GNet-pose](https://github.com/Guanghan/GNet-pose). Many thanks!

About comprehensive readme:

- [code.pdf](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/caffe_code/code.pdf)
 provides details about custom layers.

- [data.pdf](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/data.pdf) provides details about data format etc.

- [prototxt.pdf](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/prototxt.pdf) provides training/testing pipeline about the configuration prototxt file.

----
## Environment
- Ubuntu / Windows

For Ubuntu, I used **two 12 GB TITAN Xp**. For Windows, I used one **8 GB GTX 1070** on **Alienware Laptop**. 

- SSD

You'll need SSD for online data loading. 

----
## General Structure

   ```
   ${POSE_ROOT}
   +-- caffe_code
   +-- data
   +-- figs
   +-- models
   +-- training
   +-- testing
   +-- README.md
   ```

----
## Installation
1. install `Caffe` from [GNet Caffe repository](https://github.com/Guanghan/GNet-caffe). 
2. I have developed a myriad of layers. Code structure is

``` shell
${POSE_ROOT}
|-- caffe_code
`-- |-- include
    `-- |-- caffe
        |   |-- deep_human_model_layers.hpp
        |   |   | ### This includes operations about 2d/3d heatmap /integral / augmentation / local <-> global transformation etc.
        |   |-- h36m.h
        |   |   | ### This includes definitions of joint / part / bone (h36m 32 joints / usable 16 joints / c2f 17 joints etc.)
        |   |-- operations.hpp 
        |   |   | ### This includes operations w.r.t scalar / vector / fetch file / output data.
`-- |-- src
    `-- |-- caffe
        |   |-- layers
        |   |   |-- DeepHumanModel
        |   |   |   |-- deep_human_model_argmax_2d_hm_layer.cpp 
        |   |   |   |-- ### This takes argmax operation on 2d heatmap 
        |   |   |   |-- deep_human_model_convert_2d_layer.cpp 
        |   |   |   |-- ### h36m provides full 32 joints, of which we only care 16 joints. Conversion from 16x2 <-> 32x2
        |   |   |   |-- deep_human_model_convert_3d_layer.cpp 
        |   |   |   |-- ### Conversion from 16x3 <-> 32x3
        |   |   |   |-- deep_human_model_convert_depth_layer.cpp 
        |   |   |   |-- ### Conversion from root-relative camera coordinate <-> [-1, 1] normalized depth
        |   |   |   |-- deep_human_model_gen_3d_heatmap_in_more_detail_v3_layer.cpp 
        |   |   |   |-- ### Generate groud truth for 3d heatmap. Closely follows c2f Torch code.
        |   |   |   |-- deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_layer.cpp 
        |   |   |   |-- ### Argmax operation on 3d heatmap
        |   |   |   |-- deep_human_model_h36m_gen_aug_3d_layer.cpp 
        |   |   |   |-- ### Generate augmented 3d ground truth according to augmented 2d gt and 3d gt
        |   |   |   |-- deep_human_model_h36m_gen_pred_mono_3d_layer.cpp 
        |   |   |   |-- ### 2.5D -> 3D camera frame coordinate
        |   |   |   |-- deep_human_model_integral_vector_layer.cpp 
        |   |   |   |-- ### \sum_{i=0}^{D-1} probability * position
        |   |   |   |-- deep_human_model_integral_x_layer.cpp 
        |   |   |   |-- ### Integral along X axis
        |   |   |   |-- deep_human_model_integral_y_layer.cpp 
        |   |   |   |-- ### Integral along Y axis
        |   |   |   |-- deep_human_model_integral_z_layer.cpp 
        |   |   |   |-- ### Integral along Z axis
        |   |   |   |-- deep_human_model_norm_3d_hm_layer.cpp 
        |   |   |   |-- ### Normalize 3D heatmap responses to make them sum up to 1.0
        |   |   |   |-- deep_human_model_normalization_response_v0_layer.cpp 
        |   |   |   |-- ### 2D heatmap normalization
        |   |   |   |-- deep_human_model_numerical_coordinate_regression_layer.cpp 
        |   |   |   |-- ### Integral over normalized 2D heatmap -> (x, y)
        |   |   |   |-- deep_human_model_output_heatmap_sep_channel_layer.cpp 
        |   |   |   |-- ### Output heatmap of different joints to different folders
        |   |   |   |-- deep_human_model_output_joint_on_skeleton_map_h36m_layer.cpp 
        |   |   |   |-- ### Plot predicted joints on raw image
        |   |   |   |-- deep_human_model_softmax_3d_hm_layer.cpp 
        |   |   |   |-- ### Softmax normalization on 3d heatmap
        |   |   |   |-- deep_human_model_softmax_hm_layer.cpp 
        |   |   |   |-- ### Softmax normalization on 2d heatmap
		
		
		
        |   |   |-- Operations
		
		
		
		
        |   |   |   |-- adaptive_weight_euc_loss_layer.cpp
        |   |   |   |-- ### Adaptive weight controlling on different euclidean regression loss
        |   |   |   |-- add_vector_by_constant_layer.cpp
        |   |   |   |-- ### Add each element of vector by a scalar 
        |   |   |   |-- add_vector_by_single_vector_layer.cpp
        |   |   |   |-- ### Add two vectors element-wisely
        |   |   |   |-- add_vector_by_constant_layer.cpp
        |   |   |   |-- ### Add each element of vector by a scalar 
        |   |   |   |-- cross_validation_random_choose_index_layer.cpp
        |   |   |   |-- ### Select an index from different training split sources
        |   |   |   |-- gen_heatmap_all_channels_layer.cpp
        |   |   |   |-- ### Generate 2d heatmap ground truth. Closely follows Yichen Wei simple baseline & CPM caffe CPMDataLayer
        |   |   |   |-- gen_rand_index_layer.cpp
        |   |   |   |-- ### Randomly generate a index for training/testing
        |   |   |   |-- gen_sequential_index_layer.cpp
        |   |   |   |-- ### Sequentially generate index for testing
        |   |   |   |-- gen_unified_data_and_label_layer.cpp
        |   |   |   |-- ### Generate augmentend training data and label (2D). Adapated from CPMDataLayer  
        |   |   |   |-- joint_3d_square_root_loss_layer.cpp
        |   |   |   |-- ### Display average joint error MPJPE (mm)
        |   |   |   |-- js_regularization_loss_layer.cpp
        |   |   |   |-- ### Jenson-Shannon regularization loss
        |   |   |   |-- mul_rgb_layer.cpp
        |   |   |   |-- ### Scale rgb image by a scalar
        |   |   |   |-- output_blob_layer.cpp
        |   |   |   |-- ### Output blob to files for debugging 
        |   |   |   |-- output_heatmap_one_channel_layer.cpp
        |   |   |   |-- ### Output heatmap of one specific joint to file
        |   |   |   |-- read_blob_from_file_indexing_layer.cpp
        |   |   |   |-- ### Read data from disk w/ file index (id)
        |   |   |   |-- read_blob_from_file_layer.cpp
        |   |   |   |-- ### Read blob from a specific file
        |   |   |   |-- read_image_from_file_name_layer.cpp
        |   |   |   |-- ### Read image from file path
        |   |   |   |-- read_image_from_image_path_file_layer.cpp
        |   |   |   |-- ### Read image from a single file describing path for all images in the set
        |   |   |   |-- read_image_layer.cpp
        |   |   |   |-- ### See code
        |   |   |   |-- read_index_layer.cpp
        |   |   |   |-- ### Read image index from file
        |   |   |   |-- scale_vector_layer.cpp
        |   |   |   |-- ### Multiply vector by a constant scalar
       
```

3. Copy ```${POSE_ROOT}/caffe_code/include/caffe/*``` to ```${CAFFE_ROOT}/include/caffe/```

4. Copy ```${POSE_ROOT}/caffe_code/src/caffe/layers/*``` to ```${CAFFE_ROOT}/src/caffe/layers/``` after running the following
```
cd ${CAFFE_ROOT}src/caffe/layers
mkdir DeepHumanModel
mkdir Operations
```

5. Configure ```caffe.proto```
   - Add contents in **LayerParameter** of ```${POSE_ROOT}/caffe_code/src/caffe/proto/custom_layers_mine.proto``` to ```${CAFFE_ROOT}/src/caffe/proto/caffe.proto```

   - Replace **TransformationParameter** in ```${CAFFE_ROOT}/src/caffe/proto/caffe.proto``` with the one in mine ```${POSE_ROOT}/caffe_code/src/caffe/proto/custom_layers_mine.proto```
   
   - Add other layer parameter fields in ```${POSE_ROOT}/caffe_code/src/caffe/proto/custom_layers_mine.proto``` to ```${CAFFE_ROOT}/src/caffe/proto/caffe.proto``` 
   
   - *Make sure ID of **LayerParameter** do not conflict with each other.*

6. Compile
   ```
   sudo make all -j128
   ```
   
   **Note 1:** For ubuntu, you will have to modify header section of **gen_unified_data_and_label_layer.cpp** like this 
   ```
   #ifdef USE_OPENCV
   #include <opencv2/core/core.hpp>
   //#include <opencv2/opencv.hpp>
   //#include <opencv2/contrib/contrib.hpp>
   #include <opencv2/highgui/highgui.hpp>
   #endif  // USE_OPENCV
   ```
   
   **Note 2:** For windows, you will have to modify header section of **gen_unified_data_and_label_layer.cpp** like this 
   ```
   #ifdef USE_OPENCV
   #include <opencv2/core/core.hpp>
   #include <opencv2/opencv.hpp>
   #include <opencv2/contrib/contrib.hpp>
   #include <opencv2/highgui/highgui.hpp>
   #endif  // USE_OPENCV
   ```
   
Still can't compile? Contact me.

----
## Data
- One thing I have realized over the years is that **HDF5**, **LMDB**, **JSON**, **tar.gz**, **pth.tar** or whatever is totally redundant, and suffers from a major downside: it needs to be loaded into memory. For python-based framework e.g. Keras, it is time consuming (sometimes 30 seconds ++) to load offline data. Even for caffe, it takes several seconds. I have thus far switched to a simple and naive data format i.e. **txt**. Each txt represents an annotation for a sample e.g. ground truth 3d, bbx. SSD is required.

- See [data.pdf](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/data.pdf)  for a thorough discussion and joint definition. (full 32 joints vs usable 16 joints)

| Folder Name | Download Link | Description | A Toy Example |
|:-:|:-:|:-:|:-:|
| bbx_all_new | [bbx](https://drive.google.com/file/d/1TsPBkrkITMK2Snzb0WcUzOSVz3EIjr1p/view?usp=sharing) | (bbx_x1, bbx_y1, bbx_x2, bbx_y2) | [bbx](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/bbx_all_new/9.txt) |
| center_x | [center_x](https://drive.google.com/open?id=16iOGkEmoRU93wmHKxGpVjHvZpsVjKXra) | center_x (constant: 112.0) | [center_x](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/center_x/18.txt) |
| center_y | [center_y](https://drive.google.com/open?id=1Y-GrOKnd6V-KgnP15D7Y_PslqiDu5R40) | center_y (constant: 112.0) | [center_y](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/center_y/13.txt) |
| scale | [scale](https://drive.google.com/open?id=180JpQjGYdAN6ggoyZoH6ddwC3GxsN_py) | person image scale (constant: 224.0) | [scale](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/scale/23.txt) |
| gt_joint_2d_raw_new | [gt_2d](https://drive.google.com/open?id=1QTVJI4IntxPer1kx-jzB2PclAQMekhEr) | 2d gt on 224x224 cropped image (32x2) | [gt_2d](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/gt_joint_2d_raw_new/1000.txt) |
| image_path_file |  | image path for each sample | [img_path_file](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/image_path_file/20.txt) |
| gt_joint_3d_mono_raw | [gt_3d](https://drive.google.com/open?id=1EI8AKCorNqXPSvt3tjfLWW9YX1WsXSIP) | monocular 3d gt in camera coordiante (32x3) | [gt_3d](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/gt_joint_3d_mono_raw/1000006.txt)|
| camera_all | [camera](https://drive.google.com/open?id=1hIjwJdc6bAaaKgDmUuVbLrRDGW3zIKMa) | intrinsic & extrinsic camera parameters | [camera](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/camera_all/100.txt) |
| index_range | [ind_range](https://github.com/strawberryfg/c2f-3dhm-human-caffe/tree/master/data/full/index_range) | index range per (subject, action) | [ind_range](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/full/index_range/S6_Phoning_range.txt) |
| info_all | [basic_info](https://drive.google.com/open?id=1zzK9ysvdMs58gaCGvASLDxj0zXL6qzaM) | video/action name/subaction/camera id/frame id | [basic_info](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/info_all/1000019.txt) |
| images | [img](https://drive.google.com/open?id=1IwqkZBu4tZBTtNbOJc9Jduxpvcvf68xC) | all the cropped images (224x224) | [img](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/data/toy_example/images/1000.png) |



#### Download data, place to 
```  
${POSE_ROOT}
 `-- data
     `-- full
         |   |-- bbx_all_new
         |   |-- center_x
         |   |-- center_y
         |   |-- scale
         |   |-- gt_joint_2d_raw_new
         |   |-- gt_joint_3d_mono_raw
         |   |-- image_path_file
         |   |-- camera_all
         |   |-- index_range
         |   |-- info_all
         |   |-- images
```

#### Train Index:

``` 0 - 1559571 ```  

#### Test Index: 

``` 1559572 - 2108570 ```  

## Trained models
| Method |d2   |  MPJPE(mm)  | Caffe Model  | Solver State |
|:-:|:-:|:-:|:-:|:-:|
| Mine     | 64 | [67.1](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/figs/test_d64_full.png)    | [Google Drive (net_iter_720929.caffemodel)](https://drive.google.com/file/d/13-rF6drfrEyuiF4u-UtqycbLm6Msc8RP/view?usp=sharing) | [Google Drive (net_iter_720929.solverstate)](https://drive.google.com/file/d/1t7pkS88-8IqUAxIYcbUKROfFTrPnhLfr/view?usp=sharing) |
| Mine     | 32 | [68.6](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/figs/test_d32_full.png) | [Google Drive (net_iter_640000.caffemodel)](https://drive.google.com/file/d/1Q5ztDnossLMKoaZEwTsjdLYvK_BJDTrW/view?usp=sharing) | [Google Drive (net_iter_640000.solverstate)](https://drive.google.com/file/d/1rdxfEOwVngvyBXPSYh_1xPUPlmBOV2hS/view?usp=sharing) |
| Mine     | 16 | [73.6](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/figs/test_d16_full.png) | [Google Drive (net_iter_560000.caffemodel)](https://drive.google.com/open?id=1-sUW2vGWtgeZxUqUznSvlsAQW3e6DiQz) | [Google Drive (net_iter_560000.solverstate)](https://drive.google.com/file/d/1XmzV7HuIdMEkeRX-fY0dfXypvxrFyoGB/view?usp=sharing) |
| [C2F](https://arxiv.org/pdf/1611.07828.pdf)     | 64 | 69.8 | None | None |
| [Integral](https://arxiv.org/pdf/1711.08229.pdf) | 64 | 68.0 | None | None |

[C2F](https://arxiv.org/pdf/1611.07828.pdf) and [Integral](https://arxiv.org/pdf/1711.08229.pdf)  are Included for reference.



#### Download models, place to 
```  
${POSE_ROOT}
 `-- models
     |   |-- net_iter_560000.caffemodel 
     |   |-- net_iter_560000.solverstate
     |   |-- net_iter_640000.caffemodel 
     |   |-- net_iter_640000.solverstate 
     |   |-- net_iter_720929.caffemodel 
     |   |-- net_iter_720929.solverstate
```
----
## Kick off the testing
As you know, evaluation on the entire dataset takes time. For testing on a random subset, I implemented a random index generation layer. See screenshot **"figs/test_d64_rand.png"**, **"figs/test_d32_rand.png"**, **"figs/test_d16_rand.png"** for details.

I should claim that this is just for fun, please do not not take it seriously. You might get, say, **68.2 mm** and **68.4 mm** in two different runs.

- **d2 =  64**
  ```
  cd testing
  $CAFFE_ROOT/build/tools/caffe test -model test_d64_rand.prototxt -weights models/net_iter_720929.caffemodel -iterations 500
  ```
  This will give you **figs/rand_test_d64.png** (unstable number around **68 mm** due to small number of samples)
  
- **d2 =  32**
  ```
  $CAFFE_ROOT/build/tools/caffe test -model test_d32_rand.prototxt -weights models/net_iter_640000.caffemodel -iterations 500
  ```
  This will give you **figs/rand_test_d32.png** (unstable number around **71 mm**)
  
- **d2 =  16**
  ```
  $CAFFE_ROOT/build/tools/caffe test -model test_d16_rand.prototxt -weights models/net_iter_560000.caffemodel -iterations 500
  ```
  This will give you **figs/rand_test_d16.png** (unstable number around **74 mm**)
  
  
----
## Full testing
For full evaluation on H36M test set
 
- **d2 =  64**
  ```
  cd testing
  $CAFFE_ROOT/build/tools/caffe test -model test_d16_statsfalse.prototxt -weights models/net_iter_720929.caffemodel -iterations 183000
  ```
  This will give you **67.1 mm** (**figs/test_d64_full.png**)


- **d2 =  32**
  ```
  $CAFFE_ROOT/build/tools/caffe test -model test_d32_statsfalse.prototxt -weights models/net_iter_640000.caffemodel -iterations 183000
  ```
  This will give you **68.6 mm** (**figs/test_d32_full.png**)
 
  
- **d2 =  16**
  ```
  $CAFFE_ROOT/build/tools/caffe test -model test_d16_statsfalse.prototxt -weights models/net_iter_560000.caffemodel -iterations 183000
  ```
  This will give you **73.6 mm** (**figs/test_d16_full.png**)

## Training

Training is a bit tricky. For code structure about prototxt, see [prototxt.pdf](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/prototxt.pdf). Here's the thing:

- I started with **d2 =  2** to warm up. Simply run 
  ```
  cd training 
  $CAFFE_ROOT/build/tools/caffe train --solver=solver_d2.prototxt 
  ```
  I trained from scratch w/o MPII 2D HM pretraining, with **2.5e-5** as base_lr and **RMSProp**. 2 GPUs were used unless otherwise specified. Weight initialization is gaussian w/ **0.01 std**. Loss ratio of 3d HM to 2d HM is **0.1:1**.
  
- **d2 =  4** Finetune weights from **d2 =  2** after convergence.
  ```
  $CAFFE_ROOT/build/tools/caffe train --solver=solver_d4.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **137 mm** on train and **150 mm** on test. For eval on training set, simply uncomment **"index_lower_bound: 0" "index_upper_bound: 1559571"** of **"GenRandIndex"** layer. Loss ratio is **0.3:1**.
 
- **d2 =  8** Finetune weights from **d2 =  4** after convergence.
  ```
  $CAFFE_ROOT/build/tools/caffe train --solver=solver_d8.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **72 mm** on train and **86 mm** on test. Loss ratio is **0.1:1**.
- **d2 =  16** Finetune weights from **d2 =  8** after convergence 
  ```
  $CAFFE_ROOT/build/tools/caffe train --solver=solver_d16.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **47 mm** on train and **72 mm** on test. Loss ratio is **0.03:1**.

- **d2 =  32** Finetune weights from **d2 =  16** after net_iter_560000.solverstate 
  ```
  $CAFFE_ROOT/build/tools/caffe train --solver=solver_d32.prototxt --snapshot=net_iter_560000.solverstate 
  ```
  You will get around **39 mm** on train and **71 mm** on test. Loss ratio is **0.03:1**.
  I changed the weight initialization of 3D heatmap to normal distribution with **0.001 std** in place previous 0.01 as I found the MPJPE error did not slump.
  
- **d2 =  64** Finetune weights from **d2 =  32** after net_iter_640000.solverstate 
  ```
  $CAFFE_ROOT/build/tools/caffe train --solver=solver_d64.prototxt --snapshot=net_iter_640000.solverstate 
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
   
- I only used **L2 loss** during training. Nevertheless I have **Jenson-Shannon regularization loss**, **smooth L1 loss**, **adaptive loss**, and **integral loss** in prototxt, as can be seen in figs/*.png. Adaptive loss tries to automatically balance weight magnitude of different euclidean regression loss. See [code.pdf](https://github.com/strawberryfg/c2f-3dhm-human-caffe/blob/master/caffe_code/code.pdf) for details about integral loss.

- MPJPE error of **argmax** operation is **"error(mm)_3d_s2_max"**.

## Windows 

#### This line is just a test.

Don't excoriate windows. Mac, ubuntu, windows are all excellent operating systems.

Start cmd.exe run 
  ```
  caffe train .... 
  ```
Should you have issues installing windows caffe, contact me.

# FAQ 

Feel free to contact me at strawberryfgalois@gmail.com if you have **any** problem or suggestion.