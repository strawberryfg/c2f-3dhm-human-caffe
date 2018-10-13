# customized caffe layers
This contains usage example of all the layers. For **Input**, **Output**, **Param**, see pdf.


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