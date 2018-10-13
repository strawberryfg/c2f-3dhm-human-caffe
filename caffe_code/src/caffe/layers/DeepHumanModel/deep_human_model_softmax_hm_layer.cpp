
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"

using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHumanModelSoftmaxHMLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		
	}
	template <typename Dtype>
	void DeepHumanModelSoftmaxHMLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		gen_size_ = (bottom[0]->shape())[2];
		top_shape.push_back((bottom[0]->shape())[0]);
		channels_ = (bottom[0]->shape())[1];
		top_shape.push_back((bottom[0]->shape())[1]);
		top_shape.push_back(gen_size_);
		top_shape.push_back(gen_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelSoftmaxHMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); 

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) 
		{

			int Bid = t * channels_ * gen_size_ * gen_size_;
			int Tid = t * channels_ * gen_size_ * gen_size_;
			
			for (int channel = 0; channel < channels_; channel++)
			{
				double sum = 0.0;
				for (int row = 0; row < gen_size_; row++) 
				{
					for (int col = 0; col < gen_size_; col++) 
					{
                                           
						float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
                        sum += exp(v);
					}
				}

                for (int row = 0; row < gen_size_; row++) 
				{
				    for (int col = 0; col < gen_size_; col++) 
				    {
					    float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];					    
					    top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = exp(v) / sum;
					    
				    }
				}	
				//printf("Sum of joint %d is %12.6f\n", channel, sum);			
			}			
		}
	}

	template <typename Dtype>
	void DeepHumanModelSoftmaxHMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) 
			{
				int Bid = t * channels_ * gen_size_ * gen_size_;
				int Tid = t * channels_ * gen_size_ * gen_size_;
				for (int i = 0; i < channels_ * gen_size_ * gen_size_; i++) bottom_diff[Bid + i] = 0.0;
				for (int channel = 0; channel < channels_; channel++)
				{
					double sum = 0.0;
					for (int row = 0; row < gen_size_; row++)
					{
						for (int col = 0; col < gen_size_; col++)
						{
							float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
							sum += exp(v);
						}
					}
					//printf("Sum is %12.6f\n", sum);
					
					for (int row = 0; row < gen_size_; row++)
					{
						for (int col = 0; col < gen_size_; col++)
						{
							float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
							bottom_diff[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] += top_diff[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] * 1.0 / pow(sum, 2) * (exp(v) * sum - exp(v) * exp(v));							
						}
					}
					
				}
			}		
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelSoftmaxHMLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelSoftmaxHMLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelSoftmaxHM);
}
