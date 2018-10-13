#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"
#define maxlen 111

using namespace cv;

//#define _DEBUG
namespace caffe {



	template <typename Dtype>
	void DeepHumanModelOutputJointOnSkeletonMapH36MLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		use_raw_rgb_image_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().use_raw_rgb_image();
		show_gt_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().show_gt();
		save_path_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().save_path();
		save_size_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().save_size();
		image_source_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().image_source();
		//read the skeleton size (because the size of the skeleton map may vary?)
		skeleton_size_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().skeleton_size();
		show_skeleton_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().show_skeleton();


		//plot setting
		circle_radius_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().circle_radius();
		line_width_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().line_width();

		//is c2f joint definition (different id)
		is_c2f_ = this->layer_param_.output_joint_on_skeleton_human_h36m_param().is_c2f();
	}

	template <typename Dtype>
	void DeepHumanModelOutputJointOnSkeletonMapH36MLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{

	}

	template <typename Dtype>
	void DeepHumanModelOutputJointOnSkeletonMapH36MLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{

	}

	template <typename Dtype>
	void DeepHumanModelOutputJointOnSkeletonMapH36MLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int batSize = (bottom[0]->shape())[0];
		const Dtype* skeleton_image_data = bottom[0]->cpu_data(); //skeleton image
		const Dtype* index_data = bottom[1]->cpu_data(); //index        
		const Dtype* pred_2d_data = bottom[2]->cpu_data(); //pred 2d
		const Dtype* gt_2d_data = bottom[3]->cpu_data(); //ground truth 2d

		for (int t = 0; t < batSize; t++)
		{
			int id = index_data[t]; //index
			Mat img = Mat::zeros(Size(skeleton_size_, skeleton_size_), CV_8UC3);
			int Bid = t * 3 * skeleton_size_ * skeleton_size_;

			//read raw rgb image if exists use it as background
			if (use_raw_rgb_image_)
			{


				char imagename[maxlen];
				sprintf(imagename, "%s%d%s", image_source_.c_str(), id, ".png");

				Mat raw_rgb = imread(imagename);
				resize(raw_rgb, raw_rgb, Size(skeleton_size_, skeleton_size_));
				for (int row = 0; row < skeleton_size_; row++)
				{
					for (int col = 0; col < skeleton_size_; col++)
					{
						for (int c = 0; c < 3; c++)
						{
							img.at<Vec3b>(row, col)[c] = raw_rgb.at<Vec3b>(row, col)[c];
						}
					}
				}
			}

			//load skeleton image data
#ifdef _DEBUG
			cout<<t<<" "<<"load skeleton image data"<<"\n";
#endif
			for (int row = 0; row < skeleton_size_; row++)
			{
				for (int col = 0; col < skeleton_size_; col++)
				{
					for (int c = 0; c < 3; c++)
					{
						if (skeleton_image_data[Bid + c * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col] > 0)
						{
							if (show_skeleton_) img.at<Vec3b>(row, col)[c] = skeleton_image_data[Bid + c * skeleton_size_ * skeleton_size_ + row * skeleton_size_ + col];
						}

					}
				}
			}


			//visualize on skeleton map
#ifdef _DEBUG
			cout<<t<<" "<<"visualize on skeleton map"<<"\n";
#endif

			if (!is_c2f_)
			{
				for (int i = 0; i < JointNumPart_h36m; i++)
				{
					int Bid = t * JointNumPart_h36m * 2;
					circle(img, Point2d(pred_2d_data[Bid + i * 2] * skeleton_size_, pred_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_gt_joint_part_h36m[i][0], color_gt_joint_part_h36m[i][1], color_gt_joint_part_h36m[i][2]), circle_radius_);
					if (show_gt_) circle(img, Point2d(gt_2d_data[Bid + i * 2] * skeleton_size_, gt_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(143, 62, 255), circle_radius_);
				}
			}
			else
			{
				for (int i = 0; i < JointNum_c2f; i++)
				{
					int Bid = t * JointNum_c2f * 2;
					circle(img, Point2d(pred_2d_data[Bid + i * 2] * skeleton_size_, pred_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(color_gt_joint_c2f[i][0], color_gt_joint_c2f[i][1], color_gt_joint_c2f[i][2]), circle_radius_);
					if (show_gt_) circle(img, Point2d(gt_2d_data[Bid + i * 2] * skeleton_size_, gt_2d_data[Bid + i * 2 + 1] * skeleton_size_), 5, Scalar(143, 62, 255), circle_radius_);
				}
			}
#ifdef _DEBUG
			cout<<t<<" "<<"connecting edges"<<"\n";
#endif
			if (!is_c2f_)
			{
				for (int i = 0; i < BoneNumPart_h36m; i++)
				{
					int Bid = t * JointNumPart_h36m * 2;
					line(img, Point2d(pred_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][0]] * 2] * skeleton_size_,
						pred_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][0]] * 2 + 1] * skeleton_size_),
						Point2d(pred_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][1]] * 2] * skeleton_size_,
						pred_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][1]] * 2 + 1] * skeleton_size_),
						Scalar(color_pred_bone_part_h36m[i][0], color_pred_bone_part_h36m[i][1], color_pred_bone_part_h36m[i][2]), line_width_);
					if (show_gt_) line(img, Point2d(gt_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][0]] * 2] * skeleton_size_,
						gt_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][0]] * 2 + 1] * skeleton_size_),
						Point2d(gt_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][1]] * 2] * skeleton_size_,
						gt_2d_data[Bid + index_joint_in_part[bones_part_h36m[i][1]] * 2 + 1] * skeleton_size_),
						Scalar(color_pred_bone_part_h36m[i][0], color_pred_bone_part_h36m[i][1], color_pred_bone_part_h36m[i][2]), line_width_);

				}
			}
			else
			{
				for (int i = 0; i < BoneNum_c2f; i++)
				{
					int Bid = t * JointNum_c2f * 2;
					line(img, Point2d(pred_2d_data[Bid + bones_c2f[i][0] * 2] * skeleton_size_,
						pred_2d_data[Bid + bones_c2f[i][0] * 2 + 1] * skeleton_size_),
						Point2d(pred_2d_data[Bid + bones_c2f[i][1] * 2] * skeleton_size_,
						pred_2d_data[Bid + bones_c2f[i][1] * 2 + 1] * skeleton_size_),
						Scalar(color_pred_bone_c2f[i][0], color_pred_bone_c2f[i][1], color_pred_bone_c2f[i][2]), line_width_);
					if (show_gt_) line(img, Point2d(gt_2d_data[Bid + bones_c2f[i][0] * 2] * skeleton_size_,
						gt_2d_data[Bid + bones_c2f[i][0] * 2 + 1] * skeleton_size_),
						Point2d(gt_2d_data[Bid + bones_c2f[i][1] * 2] * skeleton_size_,
						gt_2d_data[Bid + bones_c2f[i][1] * 2 + 1] * skeleton_size_),
						Scalar(color_pred_bone_c2f[i][0], color_pred_bone_c2f[i][1], color_pred_bone_c2f[i][2]), line_width_);

				}
			}

			
#ifdef _DEBUG
			imshow("",img);
			waitKey(0); 
			cout<<t<<" "<<"resize"<<"\n";
#endif
			resize(img, img, Size(save_size_, save_size_));

			char filename[maxlen];
			//save image to folder
			sprintf(filename, "%s%d%s", save_path_.c_str(), id, ".png");
			imwrite(filename, img);

		}
	}



#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelOutputJointOnSkeletonMapH36MLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelOutputJointOnSkeletonMapH36MLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelOutputJointOnSkeletonMapH36M);


}