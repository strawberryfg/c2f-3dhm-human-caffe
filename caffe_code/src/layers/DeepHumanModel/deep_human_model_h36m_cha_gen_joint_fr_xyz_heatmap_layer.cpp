
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"

#define PI 3.14159265359
namespace caffe {

	template <typename Dtype>
	void DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_dims_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().depth_dims();
		map_size_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().map_size();
		

		x_lb_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().x_lb();
		x_ub_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().x_ub();

		y_lb_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().y_lb();
		y_ub_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().y_ub();

		z_lb_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().z_lb();
		z_ub_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().z_ub();
		joint_num_ = this->layer_param_.deep_human_model_h36m_cha_gen_joint_fr_xyz_heatmap_param().joint_num();
	}
	template <typename Dtype>
	void DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_ * 3);

		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* hm_data = bottom[0]->cpu_data(); 


		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int Tid = t * joint_num_ * 3;

			int Bid = t * joint_num_ * depth_dims_ * map_size_ * map_size_;
			for (int j = 0; j < joint_num_; j++) {
				double max_hm_value = 0.0;
				double pred_x = 0.5 * (x_ub_ - x_lb_) + x_lb_, pred_y = 0.5 * (y_ub_ - y_lb_) + y_lb_, pred_z = 0.5 * (z_ub_ - z_lb_) + z_lb_;
				for (int k = 0; k < depth_dims_; k++) {
					for (int row = 0; row < map_size_; row++) {
						for (int col = 0; col < map_size_; col++) {
							double x = double(col + 0.5) / double(map_size_) * (x_ub_ - x_lb_) + x_lb_;
							double y = double(row + 0.5) / double(map_size_) * (y_ub_ - y_lb_) + y_lb_;
							double z = double(k + 0.5) / double(depth_dims_) * (z_ub_ - z_lb_) + z_lb_;
							if (hm_data[Bid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] - max_hm_value > 1e-6)
							{
								max_hm_value = hm_data[Bid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col];
								pred_x = x;
								pred_y = y;
								pred_z = z;
							}
						}
					}
				}
				
				top_data[Tid + j * 3 + 0] = pred_x;
				top_data[Tid + j * 3 + 1] = pred_y;
				top_data[Tid + j * 3 + 2] = pred_z;
			}
		}
	}

	template <typename Dtype>
	void DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {

		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelH36MChaGenJointFrXYZHeatmap);
}
