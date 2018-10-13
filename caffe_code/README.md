# customized caffe layers
This contains usage example of all the layers. For **Input**, **Output**, **Param**, see pdf.

You can easily find usage in prototxt by searching layer type keywords in Sublime, Notepad++ or Vim.

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
**3D -> Normalized Depth**
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