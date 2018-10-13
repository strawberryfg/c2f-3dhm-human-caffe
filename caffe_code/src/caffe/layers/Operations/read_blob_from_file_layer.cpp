#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


namespace caffe {

    template <typename Dtype>
    void ReadBlobFromFileLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        file_path_ = this->layer_param_.read_blob_from_file_param().file_path();
        num_to_read_ = this->layer_param_.read_blob_from_file_param().num_to_read();     
        batch_size_ = this->layer_param_.read_blob_from_file_param().batch_size();
    }
    template <typename Dtype>
    void ReadBlobFromFileLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back(batch_size_);
        top_shape.push_back(num_to_read_); //blob
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void ReadBlobFromFileLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        int batSize = batch_size_;
        Dtype* blob_data = top[0]->mutable_cpu_data(); //blob

        FILE *fin_blob = fopen(file_path_.c_str(), "r");
        for (int i = 0; i < num_to_read_; i++) fscanf(fin_blob, "%lf", &t_data[i]);
        fclose(fin_blob);
        
        for (int t = 0; t < batSize; t++)         
        {
            int Tid = t * num_to_read_;
            for (int i = 0; i < num_to_read_; i++) blob_data[Tid + i] = t_data[i];            
        }
    }

    template <typename Dtype>
    void ReadBlobFromFileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        
    }

#ifdef CPU_ONLY
    STUB_GPU(ReadBlobFromFileLayer);
#endif

    INSTANTIATE_CLASS(ReadBlobFromFileLayer);
    REGISTER_LAYER_CLASS(ReadBlobFromFile);
}
