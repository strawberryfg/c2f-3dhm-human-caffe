# c2f-3dhm-human-caffe
This is the caffe reimplementation of "Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose". (\emph{citation})

Screenshot of eval on test set can be found in figs/ (d = 16, 32, 64).

----
## Overture
- I write C++ faster than Python.
- I write faster C++ than Python.
- I know C++ / Caffe is not easy to understand.


----
## Your briefing
2-stage Hourglass (d1 = 1, d2 = 16/32/64) w/ batch size = 3 🌝 🌚 😈

Exquisite ResNet w/ integral coming up soon.  💪

Batch size = 1 w/ group normalization trained on windows caffe coming up soon. 💩

See Guideline.pdf for detailed description.

----
## Environment
- Ubuntu / Windows

For Ubuntu, I used two 12 GB TITAN Xp. For Windows, I used one 8 GB GTX 1070 on Alienware PC. 

----
## Data
- Download preprocessed data at ... See "Data" section in Guideline.pdf.
- Download pretrained model and snapshots at ...

----
## Kick off the testing
As you know, evaluation on the entire dataset takes time. For testing on a random subset, I implemented a random index generation layer. See screenshot "figs/test_d64_rand.png" "figs/test_d32_rand.png" "figs/test_d16_rand.png" for details.

I should claim that this is just for fun, please do not not take it seriously. You might get, say, **68.2 mm** and **68.4 mm** in two different runs.

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
  cd testing
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

## Training

Training is a bit tricky. For a comprehensive interpretation, see pdf. Here's the thing:

- I started with **d = 2** to warm up. Simply run 
  ```
  cd training 
  caffe train --solver=solver_d2.prototxt 
  ```
  I trained from scratch w/o MPII 2D HM pretraining, with 2.5e-5 as base_lr and RMSProp. 2 GPUs were used unless otherwise specified. Weight initialization is gaussian w/ 0.01 std. Loss ratio of 3d HM to 2d HM is 0.1:1
  
- **d = 4** Finetune weights from **d = 2** after convergence.
  ```
  caffe train --solver=solver_d4.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **137 mm** on train and **150 mm** on test. For eval on training set, simply uncomment **"index_lower_bound: 0" "index_upper_bound: 1559571"** of **"GenRandIndex"** layer. Loss ratio is 0.3:1
 
- **d = 8** Finetune weights from **d = 4** after convergence.
  ```
  caffe train --solver=solver_d8.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **72 mm** on train and **86 mm** on test. Loss ratio is 0.1:1.
- **d = 16** Finetune weights from **d = 8** after convergence 
  ```
  caffe train --solver=solver_d16.prototxt --snapshot=net_iter_XXX.solverstate 
  ```
  You will get around **47 mm** on train and **72 mm** on test. Loss ratio is 0.03:1.

- **d = 32** Finetune weights from **d = 16** after net_iter_560000.solverstate 
  ```
  caffe train --solver=solver_d32.prototxt --snapshot=net_iter_560000.solverstate 
  ```
  You will get around **39 mm** on train and **71 mm** on test. Loss ratio is 0.03:1.
  I changed the weight initialization of 3D heatmap to normal distribution with 0.001 std in place previous 0.01 as I found the MPJPE error did not slump.
  
- **d = 64** Finetune weights from **d = 32** after net_iter_640000.solverstate 
  ```
  caffe train --solver=solver_d64.prototxt --snapshot=net_iter_640000.solverstate 
  ```
  You will get around **37 mm** on train and **68 mm** on test. Loss ratio is 0.03:1. I again changed weight initialization of 3D heatmap from 0.001 gaussian $\rightarrow$ 0.0003. 
  
This sounds pretty sketchy, right? Another way to train this is simply train d1=1,d2=64 from scratch. Details:
\emph{missing, TO DO}

- **Notes**
1. I set global stats to false during inference due to small batch size. I cannot recall the paper that mentions it. Let me find the paper.
2. The major difference between prototxt is 
   a)
   b)