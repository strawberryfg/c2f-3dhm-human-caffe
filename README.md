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

See Guideline.pdf for detailed description.

----
## Prerequisites
- Ubuntu / Windows
- 12 GB TITAN Card / 8 GB GTX 1070 (if you want to run on Alienware PC😏)

----
## Data
- Google drive link is

----
## Quick test 
- I implemented a random index generation layer for fast testing. See screenshot "figs/rand_test_d64.png" "figs/rand_test_d32.png" "figs/rand_test_d16.png" for details.
  ```
  cd testing
  caffe test -model test_d64_rand.prototxt -weights models/net_iter_720929.caffemodel -iterations 500
  ```
  This will give you an unstable number around 68-69 mm since only 1000 samples are tested. (figs/rand_test_d64.png)
  
  ```
  caffe test -model test_d32_rand.prototxt -weights models/net_iter_640000.caffemodel -iterations 500
  ```
  This will give you figs/rand_test_d32.png
  
  ```
  caffe test -model test_d16_rand.prototxt -weights models/net_iter_560000.caffemodel -iterations 500
  ```
  This will give you figs/rand_test_d16.png
  
  
----
## Full test 
- For full evaluation on H36M test set