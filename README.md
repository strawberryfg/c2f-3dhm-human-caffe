# c2f-3dhm-human-caffe
This is the caffe reimplementation of "Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose".

Screenshot of eval on test set can be found in figs/ (d = 16, 32, 64) 

----
## Overture
- I write C++ faster than Python
- I write faster C++ than Python
- Who cares?

----
## Prerequisites
- Ubuntu / Windows
- 12 GB TITAN Card / 8 GB GTX 1070 (if you want to run on Alienware PC😏)

----
## Quick test 
- I implemented a random index generation layer for fast testing.
  ```
  cd testing
  caffe test -model test_d64.prototxt -weights net_iter_720929.caffemodel -iterations 500
  ```
  This will give you an unstable number around 68-69 mm since only 1000 samples are tested.  