#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"
#define maxlen 1111
using namespace cv;

namespace caffe {

	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		save_path_ = this->layer_param_.output_heatmap_one_channel_param().save_path();
		save_size_ = this->layer_param_.output_heatmap_one_channel_param().save_size();

		heatmap_size_ = this->layer_param_.output_heatmap_one_channel_param().heatmap_size();
	}
	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	}

	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //heatmap
		const Dtype* index_data = bottom[1]->cpu_data(); //index
		for (int t = 0; t < batSize; t++) {
			Mat img = Mat::zeros(Size(heatmap_size_, heatmap_size_), CV_8UC1);
			int Bid = t * 1 * heatmap_size_ * heatmap_size_;
			for (int row = 0; row < heatmap_size_; row++) 
			{
				for (int col = 0; col < heatmap_size_; col++) 
				{					
					img.at<uchar>(row, col) = std::min(std::max(0.0, (double)bottom_data[Bid + row * heatmap_size_ + col]), 1.0) * 255.0;					
				}
			}
			resize(img, img, Size(save_size_, save_size_));
			int id = index_data[t]; //index
			char filename[maxlen];
			sprintf(filename, "%s%d%s", save_path_.c_str(), id, ".png");
			imwrite(filename, img);
		}
	}

	template <typename Dtype>
	void OutputHeatmapOneChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(OutputHeatmapOneChannelLayer);
#endif

	INSTANTIATE_CLASS(OutputHeatmapOneChannelLayer);
	REGISTER_LAYER_CLASS(OutputHeatmapOneChannel);
}
