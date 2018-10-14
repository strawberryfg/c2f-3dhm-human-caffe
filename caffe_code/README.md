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

## DeepHumanModelOutputJointOnSkeletonMapH36M
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


## DeepHumanModelSoftmax3DHM
```
#===================now integral 3d heatmap
layer {
  type: "DeepHumanModelSoftmax3DHM"
  bottom: "heatmap2_flat_scale_reshape"
  top: "heatmap2_norm"
  name: "heatmap2_norm"
  deep_human_model_softmax_3d_hm_param {
      joint_num: 16
      depth_dims: 64
  }
}
```


## DeepHumanModelSoftmaxHM
```
#=========softmax 
layer {
  type: "DeepHumanModelSoftmaxHM"
  bottom: "heatmap2_flat_scale_reshape"
  top: "heatmap2_softmax"
  name: "heatmap2_softmax"
}
```

# Operations

## AdaptiveWeightEucLoss
```
#======adaptive loss
layer {
  name: "ada_loss"
  type: "AdaptiveWeightEucLoss"
  top: "ada_loss"
  bottom: "heatmap_flatten"
  bottom: "label_2dhm_flatten"
  bottom: "heatmap2_flatten"
  bottom: "label_3dhm_flatten"
  loss_weight: 0.0
}
```


## AddVectorByConstant
```
layer {
 type: "AddVectorByConstant"
 bottom: "img_mul_flatten"
 top: "img_mul_flatten_add"
 name: "img_mul_flatten_add"
 add_vector_by_constant_param {
    add_value: 128.0
 }
}
```

## AddVectorBySingleVector
```
layer {
 type: "AddVectorBySingleVector"
 bottom: "vec_a"
 bottom: "vec_b"
 top: "vec_aplusb"
 name: "add_vec_a_vec_b_ele_wise_ly"
}
```




## CrossValidationRandomChooseIndex
```
#=======concat index from different data sources for randomly selection
layer {
  type: "Concat"
  bottom: "image_index_h36m"
  bottom: "image_index_mpii"
  bottom: "image_index_lsp"
  top: "image_index_concat"
  include {
     phase: TRAIN
  }
}

#========randomly choose an index from concatenated vector
layer {
  type: "CrossValidationRandomChooseIndex"
  bottom: "image_index_concat"
  top: "image_index"
  name: "image_index"
  include {
     phase: TRAIN
  }
}
```


## GenHeatmapAllChannels
```
#=======generate 2d heatmap ground truth
layer {
  type: "GenHeatmapAllChannels"
  bottom: "crop_gt_joint_2d"
  top: "label_2dhm"
  name: "label_2dhm"
  gen_heatmap_all_channels_param {
       gen_size: 64
       all_one: false
       joint_num: 16
       use_baseline_render: true
       use_cpm_render: false
       stride: 4
       render_sigma: 2
  }
}
```



## GenRandIndex
```
#-----randomly generate index for h36m
layer {
  name: "gen_rand_ind"
  type: "GenRandIndex"
  top: "image_index"
  gen_rand_index_param {
     index_lower_bound: 0
     index_upper_bound: 1559571
     batch_size: 2
  }
  include {
     phase: TRAIN
  }
}
```

**For testing on train, uncomment line 3-4 and comment line 1-2** in **gen_rand_index_param** section

```
#-----randomly generate index for h36m
layer {
  name: "gen_rand_ind"
  type: "GenRandIndex"
  top: "image_index"
  gen_rand_index_param {
     index_lower_bound: 1559572
     index_upper_bound: 2108570
     #index_lower_bound: 0
     #index_upper_bound: 1559571
     batch_size: 2
  }
  include {
     phase: TEST
  }
}
```


## GenSequentialIndex
```
layer {
  type: "GenSequentialIndex"
  top: "image_index"
  gen_sequential_index_param {
      batch_size: 3
      current_index_file_path: "cur_id_test_d64_statsfalse.txt"
      num_of_samples: 548999
      start_index: 1559572
  }
  include {
      phase: TEST
  }
}
```





## GenUnifiedDataAndLabel
**The case for training**
```
#=======generate unified augmented image and augmented 2d annotation (name gt joint 2d raw H36M all is a bit tricky)
layer {
  type: "GenUnifiedDataAndLabel"
  bottom: "image_index"
  bottom: "center_x"
  bottom: "center_y"
  bottom: "scale_provided"
  bottom: "gt_joint_2d_raw_H36M"
  top: "image"
  top: "crop_gt_joint_2d_scale" 
  transform_param {
    stride: 4
    max_rotate_degree: 40.0
    crop_size_x: 256
    crop_size_y: 256
    scale_prob: 1.0
    scale_min: 0.699999988079
    scale_max: 1.29999995232
    #target_dist: 1.17100000381
    target_dist: 1.0
    center_perterb_max: 0.0
    do_clahe: false
    put_gaussian: false
    file_name_file_prefix:  "/data/wqf/h36m/mine/Human3.6M/image_path_file/"
    transform_body_joint: true
    num_parts: 16
    minus_pixel_value: 128.0
  }
  include {
    phase: TRAIN
  }
}
```


**The case for testing**
```
#=======generate unified augmented image and augmented 2d annotation (name gt joint 2d raw H36M all is a bit tricky)
layer {
  type: "GenUnifiedDataAndLabel"
  bottom: "image_index"
  bottom: "center_x"
  bottom: "center_y"
  bottom: "scale_provided"
  bottom: "gt_joint_2d_raw_H36M"
  top: "image"
  top: "crop_gt_joint_2d_scale" 
  transform_param {
    stride: 4
    max_rotate_degree: 0.0
    crop_size_x: 256
    crop_size_y: 256
    scale_prob: -1.0
    flip_prob: -1.0
    scale_min: 0.699999988079
    scale_max: 1.29999995232
    #target_dist: 1.17100000381
    target_dist: 1.0
    center_perterb_max: 0.0
    do_clahe: false
    put_gaussian: false
    file_name_file_prefix:  "/data/wqf/h36m/mine/Human3.6M/image_path_file/"
    transform_body_joint: true
    num_parts: 16
    minus_pixel_value: 128.0
  }
  include {
    phase: TEST
  }
}
```


## Joint3DSquareRootLoss
```
#=======MPJPE of s2 integral 3D heatmap
layer {
  name: "error(mm)_3d_s2_int"
  type: "Joint3DSquareRootLoss"
  bottom: "pred_joint_global_s2_int"
  bottom: "aug_3d"
  top: "error(mm)_3d_s2_int"
  loss_weight: 0.0
  joint_3d_square_root_loss_param {
     joint_num: 16
  }
}
```

## JSRegularizationLoss
```
layer {
  type: "JSRegularizationLoss"
  bottom: "heatmap"
  bottom: "label_2dhm"
  name: "js_1_2dhm"
  top: "js_1_2dhm"
  loss_weight: 0.0
   include {
    phase: TRAIN
 }
}
```





## MulRGB
```
layer {
 type: "MulRGB"
 bottom: "image"
 top: "img_mul"
 name: "img_mul"
 mul_rgb_param {
  mul_factor: 256.0
 }
 include {
   phase: TRAIN
 }
}

layer {
 type: "Flatten"
 bottom: "img_mul"
 top: "img_mul_flatten"
 name: "img_mul_flatten"
 include {
   phase: TRAIN
 }
}


layer {
 type: "AddVectorByConstant"
 bottom: "img_mul_flatten"
 top: "img_mul_flatten_add"
 name: "img_mul_flatten_add"
 
 add_vector_by_constant_param {
  add_value: 128.0
 }
 include {
    phase: TRAIN
 }
}
```

## OutputBlob
```
layer {
  name: "output_annot_depth"
  type: "OutputBlob"
  bottom: "crop_gt_joint_2d"
  bottom: "image_index"
  output_blob_param {
     blob_name: "an"
     save_path: "g/"
  }
}
```



## OutputHeatmapOneChannel
```
layer {
  name: "output_hm_one_channel"
  type: "OutputHeatmapOneChannel"
  bottom: "pred_hm_joint_12"
  bottom: "image_index"
  output_blob_param {
     blob_name: "pred_hm"
     save_path: "pred_hm/12"
  }
}
```

## ReadBlobFromFileIndexing
```
#----- Read bbx x1 bbx y1 bbx x2 bbx y2 H36M
layer {
  name: "read_bbx_H36M"
  type: "ReadBlobFromFileIndexing"
  bottom: "image_index"
  top: "bbx_H36M"
  read_blob_from_file_indexing_param {
     file_prefix: "/data/wqf/h36m/mine/Human3.6M/bbx_all_new/"
   num_to_read: 4
  }
}
``` 

## ReadBlobFromFile 
**read a txt including only one number 112.0**
```
----- Read bbx x1 bbx y1 bbx x2 bbx y2 H36M
layer {
  name: "read_center_x"
  type: "ReadBlobFromFile"
  top: "center_x"
  read_blob_from_file_param {
     file_prefix: "/data/wqf/h36m/mine/Human3.6M/center_x_112.txt"
     num_to_read: 1
	 batch_size: 3
  }
}
```



## ReadImageFromFileName
```
#----- Read image 
layer {
  type: "ReadImageFromFileName"
  bottom: "image_index"
  top: "image_read"
  name: "image_read"
  read_image_from_file_name_param {
     resize_size: 224
	 pad_square: true
	 channel_num: 3
	 file_name_file_prefix: "D:\\humanpose\\LIP\\train\\train_id\\"
	 pad_to_a_constant_size_before_resize: false
	 pad_to_constant_size: 800
  }
  include {
     phase: TRAIN
  }
}
```