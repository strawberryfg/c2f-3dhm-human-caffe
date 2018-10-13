#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


namespace caffe {

    template <typename Dtype>
    void GenSequentialIndexLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        batch_size_ = this->layer_param_.gen_sequential_index_param().batch_size();
        current_index_file_path_ = this->layer_param_.gen_sequential_index_param().current_index_file_path();
        num_of_samples_ = this->layer_param_.gen_sequential_index_param().num_of_samples();
        start_index_ = this->layer_param_.gen_sequential_index_param().start_index();
    }
    template <typename Dtype>
    void GenSequentialIndexLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back(batch_size_);
        top_shape.push_back(1); 
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void GenSequentialIndexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        int batSize = batch_size_;
        Dtype* top_index_data = top[0]->mutable_cpu_data(); //top index

       

        for (int t = 0; t < batSize; t++) 
        {
            //read until last end
            //read current index
            FILE *fin_current_id = fopen(current_index_file_path_.c_str(), "r");
            int cur_id;
            fscanf(fin_current_id, "%d", &cur_id);
            fclose(fin_current_id);

            cur_id = (cur_id + 1) % num_of_samples_;
            FILE *fout_current_id = fopen(current_index_file_path_.c_str(), "w");
            fprintf(fout_current_id, "%d\n", cur_id);
            fclose(fout_current_id);

            int Tid = t;
            top_index_data[Tid] = cur_id + start_index_;
        }
        
    }

    template <typename Dtype>
    void GenSequentialIndexLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }

#ifdef CPU_ONLY
    STUB_GPU(GenSequentialIndexLayer);
#endif

    INSTANTIATE_CLASS(GenSequentialIndexLayer);
    REGISTER_LAYER_CLASS(GenSequentialIndex);
}
