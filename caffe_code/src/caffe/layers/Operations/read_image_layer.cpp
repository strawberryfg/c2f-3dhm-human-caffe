#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"
#define maxlen 111

using namespace cv;

namespace caffe {

    template <typename Dtype>
    void ReadImageLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        read_path_ = this->layer_param_.read_image_param().read_path();
        resize_size_ = this->layer_param_.read_image_param().resize_size();   
        zero_pad_ = this->layer_param_.read_image_param().zero_pad();

        //====Added: Aug 31, 2018
        image_suffix_ =  this->layer_param_.read_image_param().image_suffix();
    }
    template <typename Dtype>
    void ReadImageLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        vector<int> top_shape;
        top_shape.push_back((bottom[0]->shape())[0]);
        top_shape.push_back(3);
        top_shape.push_back(resize_size_);
        top_shape.push_back(resize_size_);
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void ReadImageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        int batSize = (bottom[0]->shape())[0];
        const Dtype* index_data = bottom[0]->cpu_data(); //index
        Dtype* top_data = top[0]->mutable_cpu_data();
        for (int t = 0; t < batSize; t++) 
        {
            int id = index_data[t]; //index
            char filename[maxlen];
            if (zero_pad_ == 0)
                sprintf(filename, "%s%d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 1)
                sprintf(filename, "%s%01d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 2)
                sprintf(filename, "%s%02d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 3)
                sprintf(filename, "%s%03d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 4)
                sprintf(filename, "%s%04d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 5)
                sprintf(filename, "%s%05d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 6)
                sprintf(filename, "%s%06d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 7)
                sprintf(filename, "%s%07d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 8)
                sprintf(filename, "%s%08d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 9)
                sprintf(filename, "%s%09d%s", read_path_.c_str(), id, image_suffix_.c_str());
            else if (zero_pad_ == 10)
                sprintf(filename, "%s%010d%s", read_path_.c_str(), id, image_suffix_.c_str());
            //printf("%s\n", filename);
            Mat img = imread(filename);
            resize(img, img, Size(resize_size_, resize_size_));


            //=========

            for (int row = 0; row < resize_size_; row++) 
            {
                for (int col = 0; col < resize_size_; col++) 
                {
                    for (int c = 0; c < 3; c++) 
                    {
                        int Tid = t * 3 * resize_size_ * resize_size_;                       
                        top_data[Tid + c * resize_size_ * resize_size_ + row * resize_size_ + col] = img.at<Vec3b>(row, col)[c];                        
                        //printf("%12.6f ", top_data[Tid + c * resize_size_ * resize_size_ + row * resize_size_ + col]);
                    }
                }
            }
        }
    }

    template <typename Dtype>
    void ReadImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }

#ifdef CPU_ONLY
    STUB_GPU(ReadImageLayer);
#endif

    INSTANTIATE_CLASS(ReadImageLayer);
    REGISTER_LAYER_CLASS(ReadImage);
}

