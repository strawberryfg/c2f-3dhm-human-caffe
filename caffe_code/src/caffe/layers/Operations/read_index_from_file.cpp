#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"


namespace caffe {

    template <typename Dtype>
    void ReadIndexFromFileLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        index_file_path_ = this->layer_param_.read_index_from_file_param().index_file_path();
        batch_size_ = this->layer_param_.read_index_from_file_param().batch_size();
        current_index_file_path_ = this->layer_param_.read_index_from_file_param().current_index_file_path();
        num_of_samples_ = this->layer_param_.read_index_from_file_param().num_of_samples();
    }
    template <typename Dtype>
    void ReadIndexFromFileLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        vector<int> top_shape;
        top_shape.push_back(batch_size_);
        top_shape.push_back(1); //blob
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void ReadIndexFromFileLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        int batSize = batch_size_;
        Dtype* top_index_data = top[0]->mutable_cpu_data(); //top index

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

        FILE *fin_index_array = fopen(index_file_path_.c_str(), "r");
        int read_index = -1;
        //read until the current index (at first current index is -1 so reads nothing)
        for (int i = 0; i <= cur_id; i++) {
            fscanf(fin_index_array, "%d", &read_index);
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

            fscanf(fin_index_array, "%d", &read_index);

            int Tid = t;            
            top_index_data[Tid] = read_index;
        }
        fclose(fin_index_array);
    }

    template <typename Dtype>
    void ReadIndexFromFileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }

#ifdef CPU_ONLY
    STUB_GPU(ReadIndexFromFileLayer);
#endif

    INSTANTIATE_CLASS(ReadIndexFromFileLayer);
    REGISTER_LAYER_CLASS(ReadIndexFromFile);
}
