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


----
## DeepHumanModelGen3DHeatmapInMoreDetailV3
```
#=====generate 3d heatmap ground truth
layer {
  name: "gen_3d_hm"
  bottom: "annot_depth"
  bottom: "crop_gt_joint_2d"
  top: "label_3dhm"
  type: "DeepHumanModelGen3DHeatmapInMoreDetailV3"
  deep_human_model_gen_3d_heatmap_in_more_detail_v3_param {
    depth_dims: 64
    map_size: 64
    crop_size: 256
    stride: 4
    render_sigma: 2
    x_lower_bound: 0.0
    x_upper_bound: 1.0
    y_lower_bound: 0.0
    y_upper_bound: 1.0
    z_lower_bound: 0.0
    z_upper_bound: 1.0
    joint_num: 16
  }
}
```

----
## DeepHumanModelH36MChaGenJointFrXYZHeatmap
```
#========get argmax on 3D heatmap
layer {
  type: "DeepHumanModelH36MChaGenJointFrXYZHeatmap"
  bottom: "heatmap2"
  top: "pred_joint_3d_s2_max"
  name: "pred_joint_3d_s2_max"
  deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param {
     depth_dims: 64
     map_size: 64
     x_lb: 0.0
     x_ub: 1.0
     y_lb: 0.0
     y_ub: 1.0
     z_lb: 0.0
     z_ub: 1.0
     joint_num: 16
  }
}
```



----
## DeepHumanModelH36MGenAug3D
```
#========aug 2D -> aug 3D
layer {
  type: "DeepHumanModelH36MGenAug3D"
  bottom: "crop_gt_joint_2d_scale"
  bottom: "gt_joint_3d_mono_raw"
  bottom: "bbx_x1_H36M"
  bottom: "bbx_y1_H36M"
  bottom: "bbx_x2_H36M"
  bottom: "bbx_y2_H36M"
  bottom: "image_index"
  top: "aug_3d"
  name: "aug_3d"
  gen_aug_3d_param {
      joint_num: 16
      camera_parameters_prefix: "/data/wqf/h36m/mine/Human3.6M/camera_all/"
      crop_bbx_size: 256
  }
}
```