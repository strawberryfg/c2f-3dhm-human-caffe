#include <algorithm>
#include "caffe/layer.hpp"
#include "caffe/operations.hpp"
namespace caffe {

    template <typename Dtype>
    void ScaleVectorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        scale_factor_ = this->layer_param_.scale_vector_param().scale_factor();
    }

    template <typename Dtype>
    void ScaleVectorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back((bottom[0]->shape())[0]);
        top_shape.push_back((bottom[0]->shape())[1]);
        top[0]->Reshape(top_shape);
        dim_size_ = (bottom[0]->shape())[1];
    }


    template <typename Dtype>
    void ScaleVectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int batSize = (bottom[0]->shape())[0];
        for (int t = 0; t < batSize; t++) {
            for (int i = 0; i < dim_size_; i++) {
                int Tid = t * dim_size_, Bid = t * dim_size_;
                top_data[Tid + i] = bottom_data[Bid + i] * scale_factor_;
            }
        }
    }


    template <typename Dtype>
    void ScaleVectorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {

        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

        const int batSize = (bottom[0]->shape())[0];

        for (int t = 0; t < batSize; t++) {
            for (int i = 0; i < dim_size_; i++) {
                int Tid = t * dim_size_, Bid = t * dim_size_;
                bottom_diff[Bid + i] = top_diff[Tid + i] * scale_factor_;
            }
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(ScaleVectorLayer);
#endif

    INSTANTIATE_CLASS(ScaleVectorLayer);
    REGISTER_LAYER_CLASS(ScaleVector);
}  // namespace caffe
