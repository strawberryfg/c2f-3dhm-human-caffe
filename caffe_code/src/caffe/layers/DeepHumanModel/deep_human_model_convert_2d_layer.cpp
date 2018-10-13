#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/deep_human_model_layers.hpp"
namespace caffe {

	template <typename Dtype>
	void DeepHumanModelConvert2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		
	}

	template <typename Dtype>
	void DeepHumanModelConvert2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.inner_product_param().axis());
		vector<int> top_shape = bottom[0]->shape();
		top_shape.resize(axis + 1);
		t_dim_ = (bottom[0]->shape())[1];
		if (t_dim_ == JointNumPart_h36m * 2) top_shape[axis] = JointNumAll_h36m * 2;
		else top_shape[axis] = JointNumPart_h36m * 2;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelConvert2DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();		
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int batSize = (bottom[0]->shape())[0];
		for (int t = 0; t < batSize; t++)
		{
			if (t_dim_ == JointNumPart_h36m * 2)
			{
				//part to all
				//====debug .output part
				printf("Before converting:\n");
				for (int j = 0; j < JointNumPart_h36m; j++)
				{
					int Bid = t * JointNumPart_h36m * 2;
					printf("%6.2f %6.2f ", bottom_data[Bid + j * 2], bottom_data[Bid + j * 2 + 1]);
				}
				printf("\n");


				for (int i = 0; i < JointNumAll_h36m; i++)
				{
					int id = index_joint_in_part[i];
					for (int j = 0; j < 2; j++)
					{
						int Bid = t * JointNumPart_h36m * 2;
						int Tid = t * JointNumAll_h36m * 2;
						if (id == -1) top_data[Tid + i * 2 + j] = 0.5;
						else top_data[Tid + i * 2 + j] = bottom_data[Bid + id * 2 + j];
					}
				}

				printf("After:\n");
				for (int i = 0; i < JointNumAll_h36m; i++)
				{
					int Tid = t * JointNumAll_h36m * 2;
					printf("%6.2f %6.2f ", top_data[Tid + i * 2], top_data[Tid + i * 2 + 1]);
				}
				printf("\n");
			}
			else
			{
				//all to part
				for (int i = 0; i < JointNumPart_h36m; i++)
				{
					int id = index_joint_in_all[i];
					for (int j = 0; j < 2; j++)
					{
						int Bid = t * JointNumAll_h36m * 2;
						int Tid = t * JointNumPart_h36m * 2;
						if (id == -1) top_data[Tid + i * 2 + j] = 0.5;
						else top_data[Tid + i * 2 + j] = bottom_data[Bid + id * 2 + j];
					}
				}
			}				
		}
	}


	template <typename Dtype>
	void DeepHumanModelConvert2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom)
	{


	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelConvert2DLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelConvert2DLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelConvert2D);
}  // namespace caffe

