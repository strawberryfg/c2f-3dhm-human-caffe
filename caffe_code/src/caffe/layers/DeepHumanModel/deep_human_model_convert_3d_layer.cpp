#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_human_model_layers.hpp"
namespace caffe {

	template <typename Dtype>
	void DeepHumanModelConvert3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{

	}

	template <typename Dtype>
	void DeepHumanModelConvert3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		t_dim_ = (bottom[0]->shape())[1];
		if (t_dim_ == JointNumPart_h36m * 3) top_shape[axis] = JointNumAll_h36m * 3;
		else top_shape[axis] = JointNumPart_h36m * 3;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelConvert3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++)
		{
			if (t_dim_ == JointNumPart_h36m)
			{
				//part to all
				for (int i = 0; i < JointNumAll_h36m; i++)
				{
					int id = index_joint_in_part[i];
					for (int j = 0; j < 3; j++)
					{
						int Bid = t * JointNumPart_h36m * 3;
						int Tid = t * JointNumAll_h36m * 3;
						if (id == -1) top_data[Tid + i * 3 + j] = 400.0;
						else top_data[Tid + i * 3 + j] = bottom_data[Bid + id * 3 + j];
					}
				}
			}
			else
			{
				//all to part
				for (int i = 0; i < JointNumPart_h36m; i++)
				{
					int id = index_joint_in_all[i];
					for (int j = 0; j < 3; j++)
					{
						int Bid = t * JointNumAll_h36m * 3;
						int Tid = t * JointNumPart_h36m * 3;
						if (id == -1) top_data[Tid + i * 3 + j] = 400;
						else top_data[Tid + i * 3 + j] = bottom_data[Bid + id * 3 + j];
					}
				}
			}
		}
	}


	template <typename Dtype>
	void DeepHumanModelConvert3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom)
	{


	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelConvert3DLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelConvert3DLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelConvert3D);
}  // namespace caffe

