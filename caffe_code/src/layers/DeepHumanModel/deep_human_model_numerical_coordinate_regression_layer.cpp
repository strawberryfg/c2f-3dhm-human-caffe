#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"


using namespace cv;
int joint_num_;

namespace caffe {

	template <typename Dtype>
	void DeepHumanModelNumericalCoordinateRegressionLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		
		
	}
	template <typename Dtype>
	void DeepHumanModelNumericalCoordinateRegressionLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;		
		S_ = (bottom[0]->shape())[2];
                joint_num_ = (bottom[0]->shape())[1];
		top_shape.push_back((bottom[0]->shape())[0]);		
		top_shape.push_back(joint_num_ * 2);		
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelNumericalCoordinateRegressionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* hm_data = bottom[0]->cpu_data(); //heatmap data
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int Bid = t * joint_num_ * S_ * S_;
			for (int channel = 0; channel < joint_num_; channel++) 
			{			
				double pred_x = 0.0, pred_y = 0.0;
				for (int row = 0; row < S_; row++) 
				{
					for (int col = 0; col < S_; col++) 
					{
						double x = double(col) / double(S_);
						double y = double(row) / double(S_);
						double hm_value = hm_data[Bid + channel * S_ * S_ + row * S_ + col];
						pred_x += hm_value * x;
						pred_y += hm_value * y;
					}
				}
				int Tid = t * joint_num_ * 2;
				top_data[Tid + channel * 2] = pred_x;
				top_data[Tid + channel * 2 + 1] = pred_y;
			}
		}
	}


	template <typename Dtype>
	void DeepHumanModelNumericalCoordinateRegressionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();		
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) {
				//clear
				for (int j = 0; j < joint_num_ * S_ * S_; j++) bottom_diff[t * joint_num_ * S_ * S_ + j] = 0.0;
				//hm
				for (int j = 0; j < joint_num_; j++) 
				{
					int Tid = t * joint_num_ * 2;
					for (int row = 0; row < S_; row++) 
					{
						for (int col = 0; col < S_; col++) 
						{
							double x = double(col) / double(S_);
							double y = double(row) / double(S_);
							int Bid = t * joint_num_ * S_ * S_;														
							bottom_diff[Bid + row * S_ + col] = x * top_diff[Tid + j * 2];
							bottom_diff[Bid + row * S_ + col] += y * top_diff[Tid + j * 2 + 1];							
						}
					}
				}				
			}
		}

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelNumericalCoordinateRegressionLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelNumericalCoordinateRegressionLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelNumericalCoordinateRegression);
}
