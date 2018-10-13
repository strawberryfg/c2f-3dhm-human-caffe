#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


namespace caffe {

    template <typename Dtype>
    void CrossValidationRandomChooseIndexLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        
    }
    template <typename Dtype>
    void CrossValidationRandomChooseIndexLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back((bottom[0]->shape())[0]);
        top_shape.push_back(1); //index
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void CrossValidationRandomChooseIndexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

        int batSize = (bottom[0]->shape())[0];
        const Dtype* bottom_index_data = bottom[0]->cpu_data();
        int nDim = (bottom[0]->shape())[1];
        Dtype* top_index_data = top[0]->mutable_cpu_data(); //index      
        srand(time(0));
        for (int t = 0; t < batSize; t++) 
        {
            int rand_index = rand() % nDim;
            top_index_data[t] = bottom_index_data[t * nDim + rand_index]; //choose the rand_index -th index from the bottom index vector
        }
    }

    template <typename Dtype>
    void CrossValidationRandomChooseIndexLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }

#ifdef CPU_ONLY
    STUB_GPU(CrossValidationRandomChooseIndexLayer);
#endif

    INSTANTIATE_CLASS(CrossValidationRandomChooseIndexLayer);
    REGISTER_LAYER_CLASS(CrossValidationRandomChooseIndex);
}

