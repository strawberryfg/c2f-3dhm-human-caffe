#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"
using namespace cv;

namespace caffe {

	template <typename Dtype>
	void MulRGBLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		mul_factor_ = this->layer_param_.mul_rgb_param().mul_factor();
	}
	template <typename Dtype>
	void MulRGBLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(3);
		top_shape.push_back((bottom[0]->shape())[2]);
		top_shape.push_back((bottom[0]->shape())[3]);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void MulRGBLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		int size = bottom[0]->shape()[2];
		const Dtype* bottom_data = bottom[0]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {

			for (int row = 0; row < size; row++) {
				for (int col = 0; col < size; col++) {
					for (int c = 0; c < 3; c++) {
						int Bid = t * 3 * size * size;
						int Tid = t * 3 * size * size;
						top_data[Tid + c * size * size + row * size + col] = bottom_data[Bid + c * size * size + row * size + col] * mul_factor_;
					}
				}
			}

		}
	}

	template <typename Dtype>
	void MulRGBLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(MulRGBLayer);
#endif

	INSTANTIATE_CLASS(MulRGBLayer);
	REGISTER_LAYER_CLASS(MulRGB);
}
