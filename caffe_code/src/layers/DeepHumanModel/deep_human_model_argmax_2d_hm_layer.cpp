
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"

using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHumanModelArgmaxHMLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		
	}
	template <typename Dtype>
	void DeepHumanModelArgmaxHMLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		gen_size_ = (bottom[0]->shape())[2];
		top_shape.push_back((bottom[0]->shape())[0]);
		channels_ = (bottom[0]->shape())[1];
		top_shape.push_back(channels_ * 2);
		
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelArgmaxHMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); 

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) 
		{

			int Bid = t * channels_ * gen_size_ * gen_size_;
			int Tid = t * channels_ * 2;
			
			for (int channel = 0; channel < channels_; channel++)
			{
				double max_v = 0.0;
				double max_x = 0.5;
				double max_y = 0.5;
				for (int row = 0; row < gen_size_; row++) 
				{
					for (int col = 0; col < gen_size_; col++) 
					{
                                           
						float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
						if (double(v) - max_v > 1e-6)
						{
							max_v = double(v);
							max_x = double(col + 0.5) / double(gen_size_);
							max_y = double(row + 0.5) / double(gen_size_);
						}
					}
				}

				top_data[Tid + channel * 2] = max_x;
				top_data[Tid + channel * 2 + 1] = max_y;
					
			}			
		}
	}

	template <typename Dtype>
	void DeepHumanModelArgmaxHMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
				
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelArgmaxHMLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelArgmaxHMLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelArgmaxHM);
}
