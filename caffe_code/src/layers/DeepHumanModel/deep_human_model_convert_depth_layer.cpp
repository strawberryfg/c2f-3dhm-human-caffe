#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_human_model_layers.hpp"

//======1. GT 3D (J * 3) -> [-1, 1] normalized depth (J)
//======2. normalized depth [-1, 1] (J) -> GT 3D (J * 3)
namespace caffe {

	template <typename Dtype>
	void DeepHumanModelConvertDepthLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		joint_num_ = this->layer_param_.convert_depth_param().joint_num();
		depth_lb_ = this->layer_param_.convert_depth_param().depth_lb();
		depth_ub_ = this->layer_param_.convert_depth_param().depth_ub();
		root_joint_id_ = this->layer_param_.convert_depth_param().root_joint_id();
	}

	template <typename Dtype>
	void DeepHumanModelConvertDepthLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		t_dim_ = (bottom[0]->shape())[1];
		//====2.
		if (t_dim_ == joint_num_ * 3) top_shape[axis] = joint_num_;
		//====1.
		else top_shape[axis] = joint_num_;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelConvertDepthLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* gt_3d_mono_data = bottom[1]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++)
		{
			if (t_dim_ == joint_num_ * 3)
			{
				//======1. 
				for (int i = 0; i < joint_num_; i++)
				{
					int Bid = t * joint_num_ * 3;
					double gt_z = bottom_data[Bid + i * 3 + 2];
					double gt_root = gt_3d_mono_data[Bid + root_joint_id_ * 3 + 2];
					double norm_z = (gt_z - gt_root - depth_lb_) / (depth_ub_ - depth_lb_);
					//====should be in the range of [-1, 1]
					int Tid = t * joint_num_;
					top_data[Tid + i] = norm_z;
				}
			}
			else
			{
				//=======2.
				for (int i = 0; i < joint_num_; i++)
				{
					int Bid = t * joint_num_;
					double norm_z = bottom_data[Bid + i];
					int Rid = t * joint_num_ * 3;
					double gt_root = gt_3d_mono_data[Rid + root_joint_id_ * 3 + 2];
					double gt_z = norm_z * (depth_ub_ - depth_lb_) + depth_lb_ + gt_root;
					int Tid = t * joint_num_;
					top_data[Tid + i] = gt_z;
				}
			}
		}
	}


	template <typename Dtype>
	void DeepHumanModelConvertDepthLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom)
	{


	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelConvertDepthLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelConvertDepthLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelConvertDepth);
}  // namespace caffe

