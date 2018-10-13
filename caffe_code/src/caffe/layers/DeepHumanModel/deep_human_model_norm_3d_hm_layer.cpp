
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"

using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHumanModelNorm3DHMLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		joint_num_ = this->layer_param_.deep_human_model_norm_3d_hm_param().joint_num();
		depth_dims_ = this->layer_param_.deep_human_model_norm_3d_hm_param().depth_dims();
		hm_threshold_ = this->layer_param_.deep_human_model_norm_3d_hm_param().hm_threshold();
	}
	template <typename Dtype>
	void DeepHumanModelNorm3DHMLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		gen_size_ = (bottom[0]->shape())[2];
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_ * depth_dims_);
		top_shape.push_back(gen_size_);
		top_shape.push_back(gen_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelNorm3DHMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++)
		{

			int Bid = t * joint_num_ * depth_dims_ * gen_size_ * gen_size_;
			int Tid = t * joint_num_ * depth_dims_ * gen_size_ * gen_size_;

			for (int j = 0; j < joint_num_; j++)
			{
				double sum = 0.0;
				//=====sum over all values of the 3D cube belonging to this joint
				for (int d = 0; d < depth_dims_; d++)
				{
					for (int row = 0; row < gen_size_; row++)
					{
						for (int col = 0; col < gen_size_; col++)
						{
							float v = bottom_data[Bid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col];
							if (v - hm_threshold_ > 1e-6)
							{
								sum += v;
							}
						}
					}
				}

				if (fabs(sum) < 1e-6)
		        {
		    	    for (int d = 0; d < depth_dims_; d++)
			        {
				        for (int row = 0; row < gen_size_; row++)
				        {
				             for (int col = 0; col < gen_size_; col++)
					         {
                       			 top_data[Tid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col] = 1.0 / double(gen_size_) / double(gen_size_) / double(depth_dims_);
					         }
					    }
					}
				}
				else 
				{
					//======exp()/ sum(exp()) over 3D cube of this joint
					for (int d = 0; d < depth_dims_; d++)
					{
						for (int row = 0; row < gen_size_; row++)
						{
							for (int col = 0; col < gen_size_; col++)
							{
								float v = bottom_data[Bid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col];
								top_data[Tid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col] = v / sum;
							}
						}
					}
				}
				
				
				//printf("Sum of joint %d is %12.6f\n", channel, sum);			
			}
		}
	}

	template <typename Dtype>
	void DeepHumanModelNorm3DHMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++)
			{
				int Bid = t * joint_num_ * depth_dims_ * gen_size_ * gen_size_;
				int Tid = t * joint_num_ * depth_dims_ * gen_size_ * gen_size_;
				for (int i = 0; i < joint_num_ * depth_dims_ * gen_size_ * gen_size_; i++) bottom_diff[Bid + i] = 0.0;
				for (int j = 0; j < joint_num_; j++)
				{
					//======sum over exp(v)
					double sum = 0.0;
					for (int d = 0; d < depth_dims_; d++)
					{
						for (int row = 0; row < gen_size_; row++)
						{
							for (int col = 0; col < gen_size_; col++)
							{
								float v = bottom_data[Bid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col];
								if (v - hm_threshold_ > 1e-6)
								{
									sum += v;
								}
								
							}
						}
						//printf("Sum is %12.6f\n", sum);
					}

					if (fabs(sum) > 1e-6)
					{
						for (int d = 0; d < depth_dims_; d++)
						{
							for (int row = 0; row < gen_size_; row++)
							{
								for (int col = 0; col < gen_size_; col++)
								{
									float v = bottom_data[Bid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col];
									bottom_diff[Bid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col] += top_diff[Tid + j * depth_dims_ * gen_size_ * gen_size_ + d * gen_size_ * gen_size_ + row * gen_size_ + col] * (sum - v) / (sum * sum);							
								}
							}
						}
						
					}

				
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelNorm3DHMLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelNorm3DHMLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelNorm3DHM);
}
