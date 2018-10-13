#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


namespace caffe {

	template <typename Dtype>
	void ReadImageFromImagePathFileLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		image_path_file_path_ = this->layer_param_.read_image_from_image_path_file_param().image_path_file_path();
		batch_size_ = this->layer_param_.read_image_from_image_path_file_param().batch_size();
		current_index_file_path_ = this->layer_param_.read_image_from_image_path_file_param().current_index_file_path();
		num_of_samples_ = this->layer_param_.read_image_from_image_path_file_param().num_of_samples();
		resize_image_size_ = this->layer_param_.read_image_from_image_path_file_param().resize_image_size();
	}
	template <typename Dtype>
	void ReadImageFromImagePathFileLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back(batch_size_);
		top_shape.push_back(3);
		top_shape.push_back(resize_image_size_);
		top_shape.push_back(resize_image_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void ReadImageFromImagePathFileLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = batch_size_;
		Dtype* top_data = top[0]->mutable_cpu_data(); 

		//read until last end
		//read current index
		FILE *fin_current_id = fopen(current_index_file_path_.c_str(), "r");
		int cur_id;
		fscanf(fin_current_id, "%d", &cur_id);
		fclose(fin_current_id);

		//-1 
		cur_id = (cur_id + 1) % num_of_samples_ - 1;
		FILE *fout_current_id = fopen(current_index_file_path_.c_str(), "w");
		fprintf(fout_current_id, "%d\n", cur_id);
		fclose(fout_current_id);

		FILE *fin_image_path_array = fopen(image_path_file_path_.c_str(), "r");
		int read_index = -1;
		//read until the current index (at first current index is -1 so reads nothing)
		for (int i = 0; i <= cur_id; i++) {
			char t_s[maxlen];
			fscanf(fin_image_path_array, "%s", t_s);
			if (i == cur_id) break;
		}

		for (int t = 0; t < batSize; t++)
		{
			//read current index
			FILE *fin_current_id = fopen(current_index_file_path_.c_str(), "r");
			int cur_id;
			fscanf(fin_current_id, "%d", &cur_id);
			fclose(fin_current_id);

			//add one and write to file
			cur_id = (cur_id + 1) % num_of_samples_;
			FILE *fout_current_id = fopen(current_index_file_path_.c_str(), "w");
			fprintf(fout_current_id, "%d\n", cur_id);
			fclose(fout_current_id);

			//read image path
			char t_s[maxlen];
			fscanf(fin_image_path_array, "%s", t_s);
			
			//index to read image
			Mat raw_image = imread(t_s);
			//printf("reading rows %d cols %d\n", raw_image.rows, raw_image.cols);
			resize(raw_image, raw_image, Size(resize_image_size_, resize_image_size_));
			int Tid = t * 3 * resize_image_size_ * resize_image_size_;
			for (int row = 0; row < resize_image_size_; row++)
			{
				for (int col = 0; col < resize_image_size_; col++)
				{
					for (int c = 0; c < 3; c++)
					{
						top_data[Tid + c * resize_image_size_ * resize_image_size_ + row * resize_image_size_ + col] = raw_image.at<Vec3b>(row, col)[c];
					}
				}
			}
		}
		fclose(fin_image_path_array);
	}

	template <typename Dtype>
	void ReadImageFromImagePathFileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(ReadImageFromImagePathFileLayer);
#endif

	INSTANTIATE_CLASS(ReadImageFromImagePathFileLayer);
	REGISTER_LAYER_CLASS(ReadImageFromImagePathFile);
}
