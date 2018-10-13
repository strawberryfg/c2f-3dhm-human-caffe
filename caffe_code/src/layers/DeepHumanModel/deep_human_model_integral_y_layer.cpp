
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"

using namespace cv;


//N * C * H * W -> N * H
namespace caffe {

	template <typename Dtype>
	void DeepHumanModelIntegralYLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	}
	template <typename Dtype>
	void DeepHumanModelIntegralYLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[2]);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelIntegralYLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		int C = (bottom[0]->shape())[1];
		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];
		const Dtype* bottom_data = bottom[0]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {

			int Bid = t * C * H * W;
			int Tid = t * H;
			for (int y = 0; y < H; y++) top_data[Tid + y] = 0.0;
			for (int z = 0; z < C; z++) {
				for (int x = 0; x < W; x++) {
					for (int y = 0; y < H; y++) {
						top_data[Tid + y] += bottom_data[Bid + z * H * W + y * W + x];
					}
				}
			}

		}
	}

	template <typename Dtype>
	void DeepHumanModelIntegralYLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		int C = (bottom[0]->shape())[1];
		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) {
				int Bid = t * C * H * W;
				int Tid = t * H;
				for (int z = 0; z < C; z++) {
					for (int x = 0; x < W; x++) {
						for (int y = 0; y < H; y++) {
							bottom_diff[Bid + z * H * W + y * W + x] = 0.0;
						}
					}
				}

				for (int z = 0; z < C; z++) {
					for (int x = 0; x < W; x++) {
						for (int y = 0; y < H; y++) {
							bottom_diff[Bid + z * H * W + y * W + x] += top_diff[Tid + y] * 1.0;
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelIntegralYLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelIntegralYLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelIntegralY);
}
