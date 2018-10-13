#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/operations.hpp"
#define maxlen 1111
namespace caffe {

    template <typename Dtype>
    void OutputBlobLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        save_path_ = this->layer_param_.output_blob_param().save_path();
        blob_name_ = this->layer_param_.output_blob_param().blob_name();

		if_per_section_output_ = this->layer_param_.output_blob_param().if_per_section_output();

		per_section_row_num_ = this->layer_param_.output_blob_param().per_section_row_num();
		per_section_col_num_ = this->layer_param_.output_blob_param().per_section_col_num();
    }
    template <typename Dtype>
    void OutputBlobLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    }

    template <typename Dtype>
    void OutputBlobLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        int batSize = (bottom[0]->shape())[0];
        int dimSize = (bottom[0]->shape())[1];
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* index_data = bottom[1]->cpu_data(); //index

        for (int t = 0; t < batSize; t++) {
            int id = index_data[t]; //index
            int Bid = t * dimSize;
           
            char filename[maxlen];

            //save to folder
            sprintf(filename, "%s%s%s%d%s", save_path_.c_str(), blob_name_.c_str(), "_", id, ".txt");
            FILE *fout = fopen(filename, "w");

			//judge if null
			if (fout != NULL)
			{
				for (int i = 0; i < dimSize; i++)
				{
					fprintf(fout, "%12.6f ", bottom_data[Bid + i]);
					if (if_per_section_output_)
					{
						if (i % per_section_col_num_ == 0) fprintf(fout, "\n");
						if (i % (per_section_row_num_ * per_section_col_num_) == 0) fprintf(fout, "\n"); //one section completed
					}
				}
				fprintf(fout, "\n");
				fclose(fout);
			}
        }
    }

    template <typename Dtype>
    void OutputBlobLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }

#ifdef CPU_ONLY
    STUB_GPU(OutputBlobLayer);
#endif

    INSTANTIATE_CLASS(OutputBlobLayer);
    REGISTER_LAYER_CLASS(OutputBlob);
}
