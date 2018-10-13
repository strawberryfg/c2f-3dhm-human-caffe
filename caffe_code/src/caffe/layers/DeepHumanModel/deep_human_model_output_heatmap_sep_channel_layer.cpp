
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"
#define JointNum 16
#define maxlen 1111
using namespace cv;

namespace caffe {

	template <typename Dtype>
	void DeepHumanModelOutputHeatmapSepChannelLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		save_size_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().save_size();
		heatmap_size_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().heatmap_size();
		save_path_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().save_path();
		output_joint_0_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_0();
		output_joint_1_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_1();
		output_joint_2_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_2();
		output_joint_3_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_3();
		output_joint_4_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_4();
		output_joint_5_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_5();
		output_joint_6_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_6();
		output_joint_7_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_7();
		output_joint_8_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_8();
		output_joint_9_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_9();
		output_joint_10_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_10();
		output_joint_11_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_11();
		output_joint_12_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_12();
		output_joint_13_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_13();
		output_joint_14_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_14();
		output_joint_15_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().output_joint_15();
		joint_num_ = this->layer_param_.deep_human_model_output_heatmap_sep_channel_param().joint_num();
	}
	template <typename Dtype>
	void DeepHumanModelOutputHeatmapSepChannelLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


	}

	template <typename Dtype>
	void DeepHumanModelOutputHeatmapSepChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); //heatmap 16 cahnnels
		const Dtype* index_data = bottom[1]->cpu_data(); //index

		bool output_joint_[JointNum]
			=
		{
			output_joint_0_,
			output_joint_1_,
			output_joint_2_,
			output_joint_3_,
			output_joint_4_,
			output_joint_5_,
			output_joint_6_,
			output_joint_7_,
			output_joint_8_,
			output_joint_9_,
			output_joint_10_,
			output_joint_11_,
			output_joint_12_,
			output_joint_13_,
			output_joint_14_,
			output_joint_15_
		};

		for (int t = 0; t < batSize; t++) 
		{
			for (int j = 0; j < joint_num_; j++)
			{
				if (output_joint_[j])
				{
					Mat img = Mat::zeros(Size(heatmap_size_, heatmap_size_), CV_8UC1);
					for (int row = 0; row < heatmap_size_; row++) 
					{
						for (int col = 0; col < heatmap_size_; col++) 
						{
							int Bid = t * heatmap_size_ * heatmap_size_ * joint_num_;
							double v = bottom_data[Bid + j * heatmap_size_ * heatmap_size_ + row * heatmap_size_ + col];
							//original (if v < 0 v * 255 will turn out to be 255)
							//img.at<uchar>(row, col) = v * 255;
							img.at<uchar>(row, col) = max(v, 0.0) * 255;
							//important: if v < 0 let it be at least 0
						}
					}
					resize(img, img, Size(save_size_, save_size_));
					int id = index_data[t]; //index
					char filename[maxlen];
					sprintf(filename, "%s%d%s%d%s", save_path_.c_str(), j, "/", id, ".png");
					imwrite(filename, img);
				}
			}
		}

	}

	template <typename Dtype>
	void DeepHumanModelOutputHeatmapSepChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelOutputHeatmapSepChannelLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelOutputHeatmapSepChannelLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelOutputHeatmapSepChannel);
}
