
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"

using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHumanModelIntegralVectorLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		dim_lb_ = this->layer_param_.deep_human_model_integral_vector_param().dim_lb();
		dim_ub_ = this->layer_param_.deep_human_model_integral_vector_param().dim_ub();
	}
	template <typename Dtype>
	void DeepHumanModelIntegralVectorLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(1);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelIntegralVectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		int C = (bottom[0]->shape())[1];
		
		const Dtype* bottom_data = bottom[0]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {

			int Bid = t * C;
			int Tid = t * 1;
			top_data[Tid] = 0.0;
			for (int c = 0; c < C; c++)
			{
				double position = (dim_ub_ - dim_lb_) / (C) * (c + 0.5) + dim_lb_;
				top_data[Tid] += position * bottom_data[Bid + c];
			}

		}
	}

	template <typename Dtype>
	void DeepHumanModelIntegralVectorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int C = (bottom[0]->shape())[1];
		
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) 
			{
				int Bid = t * C;
				int Tid = t * 1;
				for (int c = 0; c < C; c++) bottom_diff[Bid + c] = 0.0;
				for (int c = 0; c < C; c++) 
				{
					double position = (dim_ub_ - dim_lb_) / (C) * (c + 0.5) + dim_lb_;
					bottom_diff[Bid + c] += position * top_diff[Tid];
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelIntegralVectorLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelIntegralVectorLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelIntegralVector);
}
