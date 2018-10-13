# customized caffe layers
This contains usage example of all the layers. For **Input**, **Output**, **Param**, see pdf.

You can easily find usage in prototxt by searching layer type keywords in Sublime, Notepad++ or Vim.

To understand 2D heatmap & 3D heatmap ground truth renderer, please refer to [CPM "put_gaussian_map" function](https://github.com/shihenw/caffe/blob/d154e896b48e8fb520cb4b47af8ba10bf9403382/src/caffe/data_transformer.cpp), [C2F "drawGaussian/drawGaussian3D" function](https://github.com/geopavlakos/c2f-vol-train/blob/master/src/util/img.lua), [Simple Baseline "generate_target" function](https://github.com/Microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/JointsDataset.py)

# DeepHumanModel

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


----
## DeepHumanModelH36MGenPredMono3D
```
#======get integraled 3D global (pred 3d mono raw)
layer {
  type: "DeepHumanModelH36MGenPredMono3D"
  bottom: "pred_joint_2d_s2_int"
  #bottom: "crop_gt_joint_2d"
  bottom: "pred_depth_s2_int_global"
  #bottom: "annot_depth"
  bottom: "bbx_x1_H36M"
  bottom: "bbx_y1_H36M"
  bottom: "bbx_x2_H36M"
  bottom: "bbx_y2_H36M"
  bottom: "image_index"
  top: "pred_joint_global_s2_int"
  name: "pred_joint_global_s2_int"
  gen_pred_mono_3d_param {
     joint_num: 16
     camera_parameters_prefix: "/data/wqf/h36m/mine/Human3.6M/camera_all/"
  }
}
```


----
## DeepHumanModelIntegralVector
```
layer {
  name: "integral_x_cube_0_vaue"
  type: "DeepHumanModelIntegralVector"
  bottom: "pred_cube_0_x_vec"
  top: "pred_cube_0_x"  
  deep_human_model_integral_vector_param { 
     dim_lb: 0.0
   dim_ub: 1.0
  }
}
```

## DeepHumanModelIntegralX
```
#----integral x cube_5
layer {
  name: "integral_x_cube_5_vec"
  type: "DeepHumanModelIntegralX"
  bottom: "pred_cube_5"
  top: "pred_cube_5_x_vec"  
}
```


## DeepHumanModelIntegralY
```
#----integral y cube_10
layer {
  name: "integral_y_cube_10_vec"
  type: "DeepHumanModelIntegralY"
  bottom: "pred_cube_10"
  top: "pred_cube_10_y_vec"  
}
```


## DeepHumanModelIntegralZ
```
#----integral z cube_14
layer {
  name: "integral_x_cube_14_vec"
  type: "DeepHumanModelIntegralZ"
  bottom: "pred_cube_14"
  top: "pred_cube_14_z_vec"  
}
```


## DeepHumanModelNorm3DHM
```
#===================now integral 3d heatmap
layer {
  type: "DeepHumanModelNorm3DHM"
  bottom: "heatmap2_flat_scale_reshape"
  top: "heatmap2_norm"
  name: "heatmap2_norm"
  deep_human_model_norm_3d_hm_param {
      joint_num: 16
      depth_dims: 64
      hm_threshold: 0.003
  }
}
```


## DeepHumanModelNormalizationResponseV0
```
#normalization
layer {
  bottom: "heatmap"
  top: "heatmap_norm"
  name: "norm_2dhm"
  type: "DeepHumanModelNormalizationResponseV0"
  normalization_response_param {
     hm_threshold: 0.003
  }  
}
```


## DeepHumanModelNumericalCoordinateRegression
```
#----numerical regression
layer {
  bottom: "heatmap_norm"
  top: "pred_joint_2d_s1_int"
  type: "DeepHumanModelNumericalCoordinateRegression"
  name: "pred_joint_2d_s1_int"
}
```




## DeepHumanModelOutputHeatmapSepChannel
```
#===================now output separate channel 2d heatmap
layer {
  type: "DeepHumanModelOutputHeatmapSepChannel"
  bottom: "heatmap_2d"
  bottom: "image_index"
  name: "output_2d_hm"
  deep_human_model_output_heatmap_sep_channel_param {
      joint_num: 16
      save_size: 224
      heatmap_size: 64
      save_path: "pred_hm/"
      output_joint_7: false
  }
}
```

## DeepHumanModelOutputHeatmapSepChannel
```
layer {
  type: "DeepHumanModelOutputJointOnSkeletonMapH36M"
  bottom: "img_mul_flatten_add"
  bottom: "image_index"
  bottom: "pred_joint_2d_s2_int"
  bottom: "crop_gt_joint_2d"
  output_joint_on_skeleton_human_h36m_param {
  
   use_raw_rgb_image: false
   show_gt: false
   save_path: "s1_max/"
   save_size: 224
   skeleton_size: 256
   show_skeleton: true
   is_c2f: false
  }
}
```