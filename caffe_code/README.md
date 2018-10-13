# customized caffe layers
This contains usage example of all the layers. For **Input**, **Output**, **Param**, see pdf.

You can easily find usage in prototxt by searching layer type keywords in Sublime, Notepad++ or Vim.

To understand 2D heatmap & 3D heatmap ground truth renderer, please refer to [CPM "put_gaussian_map" function](https://github.com/shihenw/caffe/blob/d154e896b48e8fb520cb4b47af8ba10bf9403382/src/caffe/data_transformer.cpp), [C2F "drawGaussian/drawGaussian3D" function](https://github.com/geopavlakos/c2f-vol-train/blob/master/src/util/img.lua), [Simple Baseline "generate_target" function](https://github.com/Microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/JointsDataset.py)

----
## DeepHumanModelArgmaxHM

```
layer {
  type: "DeepHumanModelArgmaxHM"
  bottom: "heatmap"
  top: "pred_joint_2d_s1_max"
  name: "pred_joint_2d_s1_max"
}
```

----
## DeepHumanModelConvert2D
```
layer {
  name: "gt_joint_2d_raw_H36M"
  type: "DeepHumanModelConvert2D"
  bottom: "gt_joint_2d_raw_H36M_all"
  top: "gt_joint_2d_raw_H36M"
}
```

----
## DeepHumanModelConvert3D
```
#======= 32 -> 16
layer {
  name: "gt_joint_3d_mono_raw"
  type: "DeepHumanModelConvert3D"
  bottom: "gt_joint_3d_mono_raw_all"
  top: "gt_joint_3d_mono_raw"
}
```


----
## DeepHumanModelConvertDepth
**3D -> Normalized depth**
```
layer {
  type: "DeepHumanModelConvertDepth"
  bottom: "aug_3d"
  bottom: "gt_joint_3d_mono_raw"
  top: "annot_depth"
  name: "annot_depth"
  convert_depth_param {
      joint_num: 16
      depth_lb:  -1005.185902
      depth_ub:  980.8599
      root_joint_id: 0
  }
}
```


**Normalized depth -> unnormalized(original) depth**
```
#====== get [-1, 1] predicted depth -> real depth
layer {
  type: "DeepHumanModelConvertDepth"
  bottom: "pred_depth_s2_int"
  #bottom: "annot_depth"
  bottom: "gt_joint_3d_mono_raw"
  top: "pred_depth_s2_int_global"
  name: "pred_depth_s2_int_global"
  convert_depth_param {
     joint_num: 16
     depth_lb:  -950.637313
     depth_ub:   892.544664
     root_joint_id: 0
  }
}
```