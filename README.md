# c2f-3dhm-human-caffe
This is the caffe reimplementation of "Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose".

Screenshot of eval on test set can be found in figs/ (d = 16, 32, 64) 

----
## Overture
- I write C++ faster than Python
- I write faster C++ than Python
- Who cares?

----
## Overview
2-stage Hourglass 

Exquisite ResNet w/ integral coming up soon. 

Batch size = 1 w/ group normalization trained on windows caffe coming up soon.

See Guideline.pdf for detailed description.

----
## Environment
- Ubuntu / Windows

For Ubuntu, I use two 12 GB TITAN Xp. For Windows, I use one 8 GB GTX 1070 on Alienware PC.

----
## Data
- Download preprocessed data at ... See "Data" section in Guideline.pdf.
- Download pretrained model and snapshots at ...

----
## Quick test 
I implemented a random index generation layer for fast testing. See screenshot "figs/test_d64_rand.png" "figs/test_d32_rand.png" "figs/test_d16_rand.png" for details.

- **d = 64**
  ```
  cd testing
  caffe test -model test_d64_rand.prototxt -weights models/net_iter_720929.caffemodel -iterations 500
  ```
  This will give you figs/rand_test_d64.png (unstable number around **68 mm** due to small number of samples)
  
- **d = 32**
  ```
  caffe test -model test_d32_rand.prototxt -weights models/net_iter_640000.caffemodel -iterations 500
  ```
  This will give you figs/rand_test_d32.png (unstable number around **71 mm**)
  
- **d = 16**
  ```
  caffe test -model test_d16_rand.prototxt -weights models/net_iter_560000.caffemodel -iterations 500
  ```
  This will give you figs/rand_test_d16.png (unstable number around **74 mm**)
  
  
----
## Full test 
For full evaluation on H36M test set
 
- **d = 64**
  ```
  caffe test -model test_d16_statsfalse.prototxt -weights models/net_iter_720929.caffemodel -iterations 183000
  ```
  This will give you mm (figs/test_d64_full.png)


- **d = 32**
  ```
  caffe test -model test_d32_statsfalse.prototxt -weights models/net_iter_640000.caffemodel -iterations 183000
  ```
  This will give you mm (figs/test_d32_full.png)
 
  
- **d = 16**
  ```
  caffe test -model test_d16_statsfalse.prototxt -weights models/net_iter_560000.caffemodel -iterations 183000
  ```
  This will give you **73.6** mm (figs/test_d16_full.png)

## 