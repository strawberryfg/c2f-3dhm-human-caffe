#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#define maxlen 111

using namespace cv;

namespace caffe {

	template <typename Dtype>
	void ReadImageFromFileNameLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {		
		resize_size_ = this->layer_param_.read_image_from_file_name_param().resize_size();
		pad_square_ = this->layer_param_.read_image_from_file_name_param().pad_square();
		channel_num_ = this->layer_param_.read_image_from_file_name_param().channel_num();
		file_name_file_prefix_ = this->layer_param_.read_image_from_file_name_param().file_name_file_prefix();
		pad_to_a_constant_size_before_resize_ = this->layer_param_.read_image_from_file_name_param().pad_to_a_constant_size_before_resize();
		pad_to_constant_size_ = this->layer_param_.read_image_from_file_name_param().pad_to_constant_size();
	}
	template <typename Dtype>
	void ReadImageFromFileNameLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(channel_num_);
		top_shape.push_back(resize_size_);
		top_shape.push_back(resize_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void ReadImageFromFileNameLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* index_data = bottom[0]->cpu_data(); //index
		//read index -> index the "file name" file -> read the file name -> read the image -> pad to square

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int id = index_data[t]; //index
			char filenamefile[maxlen];
			sprintf(filenamefile, "%s%d%s", file_name_file_prefix_.c_str(), id, ".txt");
			//contains the "file name"
			FILE *fin_file_name_file = fopen(filenamefile, "r");
			char file_name[maxlen];
			fscanf(fin_file_name_file, "%s", file_name);
			//the "file name"
			fclose(fin_file_name_file);

			//index "the file name" to find the image for reading
			Mat src = imread(file_name);
			int H = src.rows, W = src.cols;
			int S = max(H, W);
			//rintf("Haha read rows %d cols %d\n", src.rows, src.cols);
			Mat dst = src.clone();
			if (pad_square_)
			{				
				dst = Mat::zeros(Size(S, S), CV_8UC3);

				//padding 
				if (H > W) {
					int lb = (H - W) / 2;
					for (int row = 0; row < H; row++) {
						for (int col = 0; col < H; col++) {
							if (col >= lb && col < lb + W) {
								if (channel_num_ == 3) {
									for (int c = 0; c < 3; c++) {
										dst.at<Vec3b>(row, col)[c] = src.at<Vec3b>(row, col - lb)[c];
									}
								} else {
									dst.at<uchar>(row, col) = src.at<uchar>(row, col - lb);
								}
							}
						}
					}
				} else {
					int lb = (W - H) / 2;
					for (int row = 0; row < W; row++) {
						for (int col = 0; col < W; col++) {
							if (row >= lb && row < lb + H) {
								if (channel_num_ == 3) {
									for (int c = 0; c < 3; c++) {
										dst.at<Vec3b>(row, col)[c] = src.at<Vec3b>(row - lb, col)[c];
									}
								} else {
									dst.at<uchar>(row, col) = src.at<uchar>(row - lb, col);
								}
							}
						}
					}
				}
			}
			else
			{
				resize(dst, dst, Size(S, S));
			}

			Mat final_img;
			//if pad to constant before reize
			if (pad_to_constant_size_)
			{
				int Pad_S = min(S, pad_to_constant_size_);
				if (channel_num_ == 3) final_img = Mat::zeros(Size(Pad_S, Pad_S), CV_8UC3);
				else final_img = Mat::zeros(Size(Pad_S, Pad_S), CV_8UC1);
				int bbx_x1 = (Pad_S - S) / 2, bbx_y1 = (Pad_S - S) / 2;
				//(Pad_S - S) / 2 - (Pad_S - S) / 2 + S -1
				for (int row = bbx_y1; row < bbx_y1 + S; row++)
				{
					for (int col = bbx_x1; col < bbx_x1 + S; col++)
					{
						if (channel_num_ == 3)
						{
							for (int c = 0; c < 3; c++)
							{
								final_img.at<Vec3b>(row, col)[c] = dst.at<Vec3b>(row - bbx_y1, col - bbx_x1)[c];
							}
						}
						else final_img.at<uchar>(row, col) = dst.at<uchar>(row - bbx_y1, col - bbx_x1);
					}
				}
			}
			else 
			{
				final_img = dst.clone();
			}

			resize(final_img, final_img, Size(resize_size_, resize_size_));

			for (int row = 0; row < resize_size_; row++) 
			{
				for (int col = 0; col < resize_size_; col++) 
				{
					if (channel_num_ == 3)
					{
						for (int c = 0; c < 3; c++) 
						{
							int Tid = t * 3 * resize_size_ * resize_size_;
							top_data[Tid + c * resize_size_ * resize_size_ + row * resize_size_ + col] = final_img.at<Vec3b>(row, col)[c];
						}
					}
					else
					{
						int Tid = t * 1 * resize_size_ * resize_size_;
						top_data[Tid + row * resize_size_ + col] = final_img.at<uchar>(row, col);
					}					
				}
			}
		}
	}

	template <typename Dtype>
	void ReadImageFromFileNameLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(ReadImageFromFileNameLayer);
#endif

	INSTANTIATE_CLASS(ReadImageFromFileNameLayer);
	REGISTER_LAYER_CLASS(ReadImageFromFileName);
}

