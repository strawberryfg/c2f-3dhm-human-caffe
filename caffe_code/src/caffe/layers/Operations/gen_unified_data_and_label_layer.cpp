// output cur_epoch

//-----opencv part
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV



#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include <iostream>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"
#include "caffe/h36m.h"
#include <sstream>

#include "caffe/data_transformer.hpp"

#include "caffe/util/math_functions.hpp"


#define maxlen 1111
using namespace caffe;  // NOLINT(build/namespaces)




namespace caffe {

	template <typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//aug param (use transform_param)
		crop_x_ = this->layer_param_.transform_param().crop_size_x();
		crop_y_ = this->layer_param_.transform_param().crop_size_y();
		joint_num_ = this->layer_param_.transform_param().num_parts();  //note here is num_parts not num_parts + 1
		file_name_file_prefix_ = this->layer_param_.transform_param().file_name_file_prefix();
		minus_pixel_value_ = this->layer_param_.transform_param().minus_pixel_value();
    
	}

	template <typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//aug image blob
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(3);
		top_shape.push_back(crop_y_);
		top_shape.push_back(crop_x_);
		top[0]->Reshape(top_shape);

		top_shape.clear();
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(2 * (joint_num_));
		top[1]->Reshape(top_shape);
	}



	template<typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, Dtype *joint_data, Dtype scale_self, Dtype *objpos_x, Dtype *objpos_y) {
		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		float scale_multiplier;
		//float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
		if (dice > this->layer_param_.transform_param().scale_prob()) {
			img_temp = img_src.clone();
			scale_multiplier = 1;
		}
		else {
			float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
			scale_multiplier = (this->layer_param_.transform_param().scale_max() - this->layer_param_.transform_param().scale_min()) * dice2 + this->layer_param_.transform_param().scale_min(); //linear shear into [scale_min, scale_max]
		}
		float scale_abs = this->layer_param_.transform_param().target_dist() / scale_self;
		float scale = scale_abs * scale_multiplier;

		//printf("Scale is %12.6f Rows: %d Cols: %d \n", scale, img_src.rows, img_src.cols);
		resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);
		//modify meta data
		*objpos_x = (*objpos_x) * scale;
		*objpos_y = (*objpos_y) * scale;

		for (int j = 0; j < joint_num_; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				joint_data[j * 2 + k] *= scale;
			}
		}
	}


	template<typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::RotatePoint(cv::Point2f& p, Mat R){
		Mat point(3, 1, CV_64FC1);
		point.at<double>(0, 0) = p.x;
		point.at<double>(1, 0) = p.y;
		point.at<double>(2, 0) = 1;
		Mat new_point = R * point;
		p.x = new_point.at<double>(0, 0);
		p.y = new_point.at<double>(1, 0);
	}


	template<typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y) {

		float degree;
		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		degree = (dice - 0.5) * 2 * this->layer_param_.transform_param().max_rotate_degree();

		Point2f center(img_src.cols / 2.0, img_src.rows / 2.0);
		Mat R = getRotationMatrix2D(center, degree, 1.0);
		Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
		// adjust transformation matrix
		R.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		R.at<double>(1, 2) += bbox.height / 2.0 - center.y;
		//LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
		//          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
		//refill the border with color 128, 128, 128 (gray)
		warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128, 128, 128));

		//adjust meta data

		Point2f objpos(*objpos_x, *objpos_y);
		RotatePoint(objpos, R);
		*objpos_x = objpos.x;
		*objpos_y = objpos.y;


		for (int j = 0; j < joint_num_; j++)
		{

			Point2f joint_j(joint_data[j * 2], joint_data[j * 2 + 1]);
			RotatePoint(joint_j, R);

			joint_data[j * 2] = joint_j.x;
			joint_data[j * 2 + 1] = joint_j.y;

		}
	}



	template<typename Dtype>
	bool GenUnifiedDataAndLabelLayer<Dtype>::onPlane(cv::Point p, Size img_size) {
		if (p.x < 0 || p.y < 0) return false;
		if (p.x >= img_size.width || p.y >= img_size.height) return false;
		return true;
	}





	template<typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y) {
		float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		int crop_x = this->layer_param_.transform_param().crop_size_x();
		int crop_y = this->layer_param_.transform_param().crop_size_y();

		float x_offset = int((dice_x - 0.5) * 2 * this->layer_param_.transform_param().center_perterb_max());
		float y_offset = int((dice_y - 0.5) * 2 * this->layer_param_.transform_param().center_perterb_max());

		//LOG(INFO) << "Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
		//LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
		Point2i center(*objpos_x + x_offset, *objpos_y + y_offset);

		int offset_left = -(center.x - (crop_x / 2));
		int offset_up = -(center.y - (crop_y / 2));
		// int to_pad_right = max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
		// int to_pad_down = max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);

		img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128, 128, 128);
		for (int i = 0; i<crop_y; i++){
			for (int j = 0; j<crop_x; j++){ //i,j on cropped
				int coord_x_on_img = center.x - crop_x / 2 + j;
				int coord_y_on_img = center.y - crop_y / 2 + i;
				if (onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
					img_dst.at<Vec3b>(i, j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);
				}
			}
		}

		//gt joint 2d in raw -> gt joint 2d in [0, 1] ground truth 2d joint in bounding box
		//  +offset.x is tantamount to - bbx_x1
		//  +offset.y is tantamount to - bbx_y1
		Point2f offset(offset_left, offset_up);
		*objpos_x = (*objpos_x) + offset.x;
		*objpos_y = (*objpos_y) + offset.y;

		for (int j = 0; j < joint_num_; j++)
		{
			joint_data[j * 2] += offset.x;
			joint_data[j * 2 + 1] += offset.y;
		}
	}




	template<typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::swapLeftRight(Dtype *joint_data) {

		//head_top_cpm,       //0
		//thorax_cpm,         //1
		//right_shoulder_cpm, //2
		//right_elbow_cpm,    //3
		//right_wrist_cpm,    //4
		//left_shoulder_cpm,  //5
		//left_elbow_cpm,     //6
		//left_wrist_cpm,     //7
		//right_hip_cpm,      //8
		//right_knee_cpm,     //9
		//right_ankle_cpm,    //10
		//left_hip_cpm,       //11
		//left_knee_cpm,      //12
		//left_ankle_cpm,     //13
		//center_cpm,         //14

		//---to do : put in .h not manually set here inside function
		//.....why use numbers not enum???
		int right[6] = {
			part_RightUpLeg,
			part_RightLeg,
			part_RightFoot,
			part_RightArm,
			part_RightForeArm,
			part_RightHand };
		int left[6] = {
			part_LeftUpLeg,
			part_LeftLeg,
			part_LeftFoot,
			part_LeftArm,
			part_LeftForeArm,
			part_LeftHand
		};
		for (int i = 0; i < 6; i++)
		{
			int ri = right[i];
			int li = left[i];
			double tmp;
			tmp = joint_data[ri * 2]; joint_data[ri * 2] = joint_data[li * 2]; joint_data[li * 2] = tmp;
			tmp = joint_data[ri * 2 + 1]; joint_data[ri * 2 + 1] = joint_data[li * 2 + 1]; joint_data[li * 2 + 1] = tmp;
		}

	}




	template<typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, Dtype *joint_data, Dtype *objpos_x) {
		bool doflip;

		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		doflip = (dice <= this->layer_param_.transform_param().flip_prob());

		if (doflip){
			flip(img_src, img_aug, 1);
			int w = img_src.cols;

			*objpos_x = w - 1 - (*objpos_x);

			for (int j = 0; j < joint_num_; j++)
			{
				joint_data[j * 2] = w - 1 - joint_data[j * 2];
			}

			if (this->layer_param_.transform_param().transform_body_joint())
				swapLeftRight(joint_data);

		}
		else {
			img_aug = img_src.clone();
		}
	}



	template <typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];

		const Dtype* index_data = bottom[0]->cpu_data(); //---different from GenH36MDataAndLabelLayer which is image_data_blob
		const Dtype* objpos_x_data = bottom[1]->cpu_data();
		const Dtype* objpos_y_data = bottom[2]->cpu_data();
		const Dtype* scale_provided_data = bottom[3]->cpu_data();
		const Dtype* gt_joint_2d_raw_data = bottom[4]->cpu_data();

		Dtype* transformed_data = top[0]->mutable_cpu_data();
		Dtype* transformed_label = top[1]->mutable_cpu_data();



		//no top data blob
		for (int t = 0; t < batSize; ++t)
		{
			Dtype img_height = h36m_height;
			Dtype img_width = h36m_width;
			Dtype objpos_x = objpos_x_data[t];
			Dtype objpos_y = objpos_y_data[t];
			Dtype scale_provided = scale_provided_data[t];
			Dtype gt_joint_2d_raw[111];

			for (int i = 0; i < joint_num_ * 2; i++) gt_joint_2d_raw[i] = gt_joint_2d_raw_data[t * joint_num_ * 2 + i];

			/*Read image from file name blob bottom[0]
			*/

			
			int id = index_data[t]; //index
			char filenamefile[maxlen];
			sprintf(filenamefile, "%s%d%s", file_name_file_prefix_.c_str(), id, ".txt");
			//contains the "file name"
			FILE *fin_file_name_file = fopen(filenamefile, "r");
			char file_name[maxlen];
			fscanf(fin_file_name_file, "%s", file_name);
			//printf("Reading image %shaha\n", file_name);
			//the "file name"
			fclose(fin_file_name_file);

			//Mat ttt = Mat::zeros(256, 256, CV_8UC3);
			//printf("TTT rows cols %d %d\n", ttt.rows, ttt.cols);

			//ttt = imread("/data/wqf/fs.jpg");
			//printf("TTT rows cols %d %d\n", ttt.rows, ttt.cols);


			Mat img_src = imread(file_name);
			//printf("Reading image %shaha\n", file_name);

			//printf("Rows and Cols of img_src is %d %d\n", img_src.rows, img_src.cols);
			//index "the file name" to find the image for reading

			/*Mat img_src = Mat::zeros(crop_y_, crop_x_, CV_8UC3);
			for (int row = 0; row < crop_y_; row++)
			{
				for (int col = 0; col < crop_x_; col++)
				{
					for (int c = 0; c < 3; c++)
					{
						int Bid = t * 3 * crop_y_ * crop_x_;
						img_src.at<Vec3b>(row, col)[c] = img_data[Bid + c * crop_y_ * crop_x_ + row * crop_x_ + col];
					}
				}
			}*/


			//Start transforming
			Mat img_aug = Mat::zeros(crop_y_, crop_x_, CV_8UC3);
			//temporary mat
			Mat img_temp, img_temp2, img_temp3; //size determined by scale


			augmentation_scale(img_src, img_temp, gt_joint_2d_raw, scale_provided, &objpos_x, &objpos_y);
			augmentation_rotate(img_temp, img_temp2, gt_joint_2d_raw, &objpos_x, &objpos_y);
			augmentation_croppad(img_temp2, img_temp3, gt_joint_2d_raw, &objpos_x, &objpos_y);
			augmentation_flip(img_temp3, img_aug, gt_joint_2d_raw, &objpos_x);


			//save aug image to top blob [0]
			int offset = img_aug.rows * img_aug.cols;
			for (int row = 0; row < img_aug.rows; row++)
			{
				for (int col = 0; col < img_aug.cols; col++)
				{
					Vec3b& rgb = img_aug.at<Vec3b>(row, col);
					int Tid = t * 3 * offset;
					transformed_data[Tid + 0 * offset + row * img_aug.cols + col] = (rgb[0] - minus_pixel_value_) / 256.0;
					transformed_data[Tid + 1 * offset + row * img_aug.cols + col] = (rgb[1] - minus_pixel_value_) / 256.0;
					transformed_data[Tid + 2 * offset + row * img_aug.cols + col] = (rgb[2] - minus_pixel_value_) / 256.0;
				}
			}

			//save aug label to top blob[1]
			// last one is center(nothing; background) can be ignored
			int Tid = t * joint_num_ * 2;
			for (int j = 0; j < 2 * (joint_num_); j++)
			{
				transformed_label[Tid + j] = 0.0;
			}
			//LOG(INFO) << "label cleaned";

			for (int j = 0; j < joint_num_; j++)
			{
				//joints is point2f 
				transformed_label[Tid + 2 * j + 0] = gt_joint_2d_raw[j * 2];
				transformed_label[Tid + 2 * j + 1] = gt_joint_2d_raw[j * 2 + 1];
			}
		}
	}

	template <typename Dtype>
	void GenUnifiedDataAndLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(GenUnifiedDataAndLabelLayer);
#endif

	INSTANTIATE_CLASS(GenUnifiedDataAndLabelLayer);
	REGISTER_LAYER_CLASS(GenUnifiedDataAndLabel);
}


