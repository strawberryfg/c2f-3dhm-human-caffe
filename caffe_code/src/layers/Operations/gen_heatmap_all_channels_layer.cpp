
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"

using namespace cv;

namespace caffe {

	template <typename Dtype>
	void GenHeatmapAllChannelsLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		gen_size_ = this->layer_param_.gen_heatmap_all_channels_param().gen_size();

		render_sigma_ = this->layer_param_.gen_heatmap_all_channels_param().render_sigma();

		all_one_ = this->layer_param_.gen_heatmap_all_channels_param().all_one();
		joint_num_ = this->layer_param_.gen_heatmap_all_channels_param().joint_num();

		use_cpm_render_ = this->layer_param_.gen_heatmap_all_channels_param().use_cpm_render();
		use_baseline_render_ = this->layer_param_.gen_heatmap_all_channels_param().use_baseline_render();
		// ===== simple baseline for human pose estimation and tracking Yichen Wei PyTorch code JointsDataset.py 
		// ===== same as coarse to fine 3d heatmap drawGaussian (2D heatmap)

		crop_size_ = this->layer_param_.gen_heatmap_all_channels_param().crop_size();
		stride_ = this->layer_param_.gen_heatmap_all_channels_param().stride();
		grid_size_ = crop_size_ / stride_;
	}
	template <typename Dtype>
	void GenHeatmapAllChannelsLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_);
		top_shape.push_back(gen_size_);
		top_shape.push_back(gen_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void GenHeatmapAllChannelsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //2d gt [0,1]

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {

			int Bid = t * joint_num_ * 2;

			int Tid = t * joint_num_ * gen_size_ * gen_size_;
			if (!use_cpm_render_ && !use_baseline_render_)
			{
				for (int row = 0; row < gen_size_; row++)
				{
					for (int col = 0; col < gen_size_; col++)
					{
						for (int channel = 0; channel < joint_num_; channel++)
						{
							float gt_x = bottom_data[Bid + channel * 2], gt_y = bottom_data[Bid + channel * 2 + 1];
							float t = exp(-1.0 / (2.0 * (render_sigma_ * render_sigma_)) * (pow(col / float(gen_size_) - gt_x, 2) + pow(row / float(gen_size_) - gt_y, 2)));
							if (all_one_)
							{
								float dist = sqrt(pow(col / float(gen_size_) - gt_x, 2) + pow(row / float(gen_size_) - gt_y, 2));
								if (render_sigma_ - dist > 1e-6)
								{
									//within radius
									top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = 1.0;
								}
								else
								{
									top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = 0.0;
								}
							}
							else
							{
								top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = t;
							}
						}
					}
				}
			}
			else if (use_cpm_render_)
			{
				for (int channel = 0; channel < joint_num_; channel++)
				{
					for (int row = 0; row < gen_size_; row++)
					{
						for (int col = 0; col < gen_size_; col++)
						{
							top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = 0.0;
						}
					}
				}
				float start = stride_ / 2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
				for (int g_y = 0; g_y < grid_size_; g_y++)
				{
					for (int g_x = 0; g_x < grid_size_; g_x++)
					{
						for (int channel = 0; channel < joint_num_; channel++)
						{
							float x = start + g_x * stride_;
							float y = start + g_y * stride_;
							float center_x = bottom_data[Bid + channel * 2] * crop_size_;
							float center_y = bottom_data[Bid + channel * 2 + 1] * crop_size_;
							float d2 = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);
							float exponent = d2 / 2.0 / render_sigma_ / render_sigma_;
							if (exponent > 4.6052)
							{ //ln(100) = -ln(1%)
								continue;
							}
							top_data[Tid + channel * gen_size_ * gen_size_ + g_y * gen_size_ + g_x] += exp(-exponent);
							if (top_data[Tid + channel * gen_size_ * gen_size_ + g_y * gen_size_ + g_x] > 1)
							{
								top_data[Tid + channel * gen_size_ * gen_size_ + g_y * gen_size_ + g_x] = 1;
							}
						}
					}
				}
			}
			else if (use_baseline_render_)
			{
				int tmp_size = render_sigma_ * 3;
				int size = 2 * tmp_size + 1;
				//=====clear array first
				for (int channel = 0; channel < joint_num_; channel++)
				{
					for (int row = 0; row < gen_size_; row++)
					{
						for (int col = 0; col < gen_size_; col++)
						{
							top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = 0.0;
						}
					}
				}

				for (int channel = 0; channel < joint_num_; channel++)
				{
					int mu_x = int(bottom_data[Bid + channel * 2] * crop_size_ / stride_ + 0.5);
					int mu_y = int(bottom_data[Bid + channel * 2 + 1] * crop_size_ / stride_ + 0.5);
					int x_lb = mu_x - tmp_size;
					int x_ub = mu_x + tmp_size;
					int y_lb = mu_y - tmp_size;
					int y_ub = mu_y + tmp_size;
					for (int row = y_lb; row <= y_ub; row++)
					{
						for (int col = x_lb; col <= x_ub; col++)
						{
							if (row >= 0 && row < gen_size_ && col >= 0 && col < gen_size_)
							{
								int cur_x = col - mu_x;
								int cur_y = row - mu_y;
								double h_v = exp(-1.0 / (2.0 * render_sigma_ * render_sigma_) * (pow(cur_x, 2) + pow(cur_y, 2)));
								top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = h_v;
							}
						}
					}
				}

			}
		}
	}

	template <typename Dtype>
	void GenHeatmapAllChannelsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {

		}
	}

#ifdef CPU_ONLY
	STUB_GPU(GenHeatmapAllChannelsLayer);
#endif

	INSTANTIATE_CLASS(GenHeatmapAllChannelsLayer);
	REGISTER_LAYER_CLASS(GenHeatmapAllChannels);
}

