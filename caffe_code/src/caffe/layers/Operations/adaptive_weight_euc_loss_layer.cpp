#include <algorithm>

#include "caffe/operations.hpp"

#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void AdaptiveWeightEucLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
        if (this->layer_param_.loss_weight_size() == 0) {
			this->layer_param_.add_loss_weight(Dtype(1));
		}

		
	}

	template <typename Dtype>
	void AdaptiveWeightEucLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape(0);
		top[0]->Reshape(loss_shape);
		num_of_losses_ = bottom.size() / 2; //====pred gt pairs
		//printf("Num of losses : %d\n", num_of_losses_);
	}


	template <typename Dtype>
	void AdaptiveWeightEucLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
		Dtype loss = 0;

		const int batSize = (bottom[0]->shape())[0];
		
		//=======Loop over all objective functions to figure out magnitude of |x_i - x_i GT|
		for (int i = 0; i < num_of_losses_; i++)
		{
			avg_abs_diff_each_dim[i] = 0.0;
			//======Get prediction and ground truth blob of this objective function
			const Dtype* pred_blob_data = bottom[2 * i]->cpu_data();
			const Dtype* gt_blob_data = bottom[2 * i + 1]->cpu_data();

			int ndim = (bottom[2 * i]->shape())[1]; //====should be equal (bottom[2 * i + 1]->shape())[1]
			//======Loop over all samples in this mini batch
			for (int t = 0; t < batSize; t++)
			{
				int Bid = t * ndim;
				for (int j = 0; j < ndim; j++)
				{
					avg_abs_diff_each_dim[i] += fabs(double(pred_blob_data[Bid + j] - double(gt_blob_data[Bid + j])));
				}
			}
			avg_abs_diff_each_dim[i] /= double(ndim * batSize);
		}

		//=======Compute loss
		for (int i = 0; i < num_of_losses_; i++)
		{
			const Dtype* pred_blob_data = bottom[2 * i]->cpu_data();
			const Dtype* gt_blob_data = bottom[2 * i + 1]->cpu_data();
			int ndim = (bottom[2 * i]->shape())[1]; //====should be equal (bottom[2 * i + 1]->shape())[1]
			double weight_loss = 1.0 / avg_abs_diff_each_dim[i]; //=====take reciprocal -> make sure weight of each loss contributes 1.0
			//======Loop over all samples in this mini batch

			//printf("prepare %d loss weight %12.7f\n", i, weight_loss);
			for (int t = 0; t < batSize; t++)
			{
				int Bid = t * ndim;
				for (int j = 0; j < ndim; j++)
				{
					loss += weight_loss * pow(double(pred_blob_data[Bid + j]) - double(gt_blob_data[Bid + j]), 2);
				}
			}
		}
		//printf("prepare done\n");
			
		loss /= batSize;
		//printf("prepare done loss is %12.6f\n", loss);
		
		top[0]->mutable_cpu_data()[0] = loss;
		//printf("prepare done\n");
		
	}


	template <typename Dtype>
	void AdaptiveWeightEucLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {


		const int batSize = (bottom[0]->shape())[0];
		
		
		Dtype top_diff = top[0]->cpu_diff()[0] / batSize;
		
		//=======Backward gradient
		for (int i = 0; i < num_of_losses_; i++)
		{
			const Dtype* pred_blob_data = bottom[2 * i]->cpu_data();
			const Dtype* gt_blob_data = bottom[2 * i + 1]->cpu_data();
			Dtype* bottom_diff = bottom[2 * i]->mutable_cpu_diff();
		
			int ndim = (bottom[2 * i]->shape())[1]; //====should be equal (bottom[2 * i + 1]->shape())[1]
			
			double weight_loss = 1.0 / avg_abs_diff_each_dim[i]; //=====take reciprocal -> make sure weight of each loss contributes 1.0
			//printf("Ada weight euc loss weight #%4d is %12.6f \n", i, weight_loss);
			//======Loop over all samples in this mini batch
			for (int t = 0; t < batSize; t++)
			{
				int Bid = t * ndim;
				for (int j = 0; j < ndim; j++)
				{
					bottom_diff[Bid + j] = weight_loss * top_diff * 2.0 * (double(pred_blob_data[Bid + j]) - double(gt_blob_data[Bid + j]));
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(AdaptiveWeightEucLossLayer);
#endif

	INSTANTIATE_CLASS(AdaptiveWeightEucLossLayer);
	REGISTER_LAYER_CLASS(AdaptiveWeightEucLoss);
}  // namespace caffe
