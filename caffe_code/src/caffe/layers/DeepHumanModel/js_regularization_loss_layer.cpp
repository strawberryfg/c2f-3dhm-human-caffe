#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"

namespace caffe {

	template <typename Dtype>
	void JSRegularizationLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
		}
		min_eps_ = this->layer_param_.js_regularization_loss_param().min_eps();
	}


	template <typename Dtype>
	void JSRegularizationLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);
		top[0]->Reshape(loss_shape);
		channel_num_ = bottom[0]->shape()[1];
	}

	template <typename Dtype>
	void JSRegularizationLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* P = bottom[0]->cpu_data();
		const Dtype* Q = bottom[1]->cpu_data();

		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];

		Dtype loss = 0;
		for (int t = 0; t < batSize; t++) {
			int Bid = t * channel_num_ * H * W;

			for (int c = 0; c < channel_num_; c++) {
				for (int row = 0; row < H; row++) {
					for (int col = 0; col < W; col++) {
						if (P[Bid + c * H * W + row * W + col] > min_eps_ && Q[Bid + c * H * W + row * W + col] > min_eps_) {
							double KL_PM = -P[Bid + c * H * W + row * W + col] * log(1.0 / 2.0 * (P[Bid + c * H * W + row * W + col] + Q[Bid + c * H * W + row * W + col]) / P[Bid + c * H * W + row * W + col]);
							double KL_QM = -Q[Bid + c * H * W + row * W + col] * log(1.0 / 2.0 * (P[Bid + c * H * W + row * W + col] + Q[Bid + c * H * W + row * W + col]) / Q[Bid + c * H * W + row * W + col]);
							loss += 1.0 / 2.0 * (KL_PM + KL_QM);
						}
					}
				}
			}


		}
		top[0]->mutable_cpu_data()[0] = loss / batSize;
	}


	template <typename Dtype>
	void JSRegularizationLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* P = bottom[0]->cpu_data();
		const Dtype* Q = bottom[1]->cpu_data();
		int H = (bottom[0]->shape())[2];
		int W = (bottom[0]->shape())[3];

		Dtype top_diff = top[0]->cpu_diff()[0] / batSize;
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) {
				int Bid = t * channel_num_ * H * W;
				for (int c = 0; c < channel_num_; c++) {
					for (int row = 0; row < H; row++) {
						for (int col = 0; col < W; col++) {
							if (P[Bid + c * H * W + row * W + col] > min_eps_ && Q[Bid + c * H * W + row * W + col] > min_eps_) {
								bottom_diff[Bid + c * H * W + row * W + col] = -1.0 / 2.0 * log(1.0 / 2.0 * (P[Bid + c * H * W + row * W + col] + Q[Bid + c * H * W + row * W + col]) / P[Bid + c * H * W + row * W + col]) * top_diff;
							} else {
								bottom_diff[Bid + c * H * W + row * W + col] = 0.0;
							}
						}
					}
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(JSRegularizationLossLayer);
#endif

	INSTANTIATE_CLASS(JSRegularizationLossLayer);
	REGISTER_LAYER_CLASS(JSRegularizationLoss);

}  // namespace caffe
