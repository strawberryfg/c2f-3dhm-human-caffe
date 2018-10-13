#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


namespace caffe {

    template <typename Dtype>
    void ReadBlobFromFileIndexingLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        file_prefix_ = this->layer_param_.read_blob_from_file_indexing_param().file_prefix();
        num_to_read_ = this->layer_param_.read_blob_from_file_indexing_param().num_to_read();       
    }
    template <typename Dtype>
    void ReadBlobFromFileIndexingLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back((bottom[0]->shape())[0]);
        top_shape.push_back(num_to_read_); //blob
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void ReadBlobFromFileIndexingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        int batSize = (bottom[0]->shape())[0];
        const Dtype* index_data = bottom[0]->cpu_data();
        Dtype* blob_data = top[0]->mutable_cpu_data(); //blob
        for (int t = 0; t < batSize; t++)         
        {
            char blobfilename[maxlen];
            sprintf(blobfilename, "%s%d%s", file_prefix_.c_str(), (int)index_data[t], ".txt");
            FILE *fin_blob = fopen(blobfilename, "r");
            int Tid = t * num_to_read_;
            for (int i = 0; i < num_to_read_; i++) fscanf(fin_blob, "%lf", &t_data[i]);
            fclose(fin_blob);
            for (int i = 0; i < num_to_read_; i++) blob_data[Tid + i] = t_data[i];            
        }
    }

    template <typename Dtype>
    void ReadBlobFromFileIndexingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }

#ifdef CPU_ONLY
    STUB_GPU(ReadBlobFromFileIndexingLayer);
#endif

    INSTANTIATE_CLASS(ReadBlobFromFileIndexingLayer);
    REGISTER_LAYER_CLASS(ReadBlobFromFileIndexing);
}
