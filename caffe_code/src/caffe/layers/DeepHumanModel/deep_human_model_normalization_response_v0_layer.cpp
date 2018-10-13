
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"

using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHumanModelNormalizationResponseV0Layer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		hm_threshold_ = this->layer_param_.normalization_response_param().hm_threshold();
		
	}
	template <typename Dtype>
	void DeepHumanModelNormalizationResponseV0Layer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		gen_size_ = (bottom[0]->shape())[2];
		top_shape.push_back((bottom[0]->shape())[0]);
		channels_ = (bottom[0]->shape())[1];
		top_shape.push_back((bottom[0]->shape())[1]);
		top_shape.push_back(gen_size_);
		top_shape.push_back(gen_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelNormalizationResponseV0Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data(); 

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) 
		{

			int Bid = t * channels_ * gen_size_ * gen_size_;
			int Tid = t * channels_ * gen_size_ * gen_size_;
			
			for (int channel = 0; channel < channels_; channel++)
			{
				double sum = 0.0;
				for (int row = 0; row < gen_size_; row++) 
				{
					for (int col = 0; col < gen_size_; col++) 
					{
                                           
						float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
                        if (v - hm_threshold_ > 1e-6) 
                        {
						     sum += v;
                        }
					}
				}

                //=====critical! important! check if all zero values 
                if (fabs(sum) < 1e-6)
		        {
		    	    for (int channel = 0; channel < channels_; channel++)
			        {
				        for (int row = 0; row < gen_size_; row++)
				        {
				             for (int col = 0; col < gen_size_; col++)
					         {
                       			 top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = 1.0 / double(gen_size_) / double(gen_size_);
					         }
					    }
					}
				}
				else  //==========not all zero pixel values
				{
                    for (int row = 0; row < gen_size_; row++) 
				    {
				   	    for (int col = 0; col < gen_size_; col++) 
					    {
						    float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
						    if (v > hm_threshold_)
						    {
						    	top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = v / sum;
						        //printf("....................%12.6f\n", v/sum);
						    }
						    else 
						    {
						    	top_data[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] = 0.0;
						    }
						    
 				        }
				    }
				}	
				//printf("Sum of joint %d is %12.6f\n", channel, sum);			
			}			
		}
	}

	template <typename Dtype>
	void DeepHumanModelNormalizationResponseV0Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) 
			{
				int Bid = t * channels_ * gen_size_ * gen_size_;
				int Tid = t * channels_ * gen_size_ * gen_size_;
				for (int i = 0; i < channels_ * gen_size_ * gen_size_; i++) bottom_diff[Bid + i] = 0.0;
				for (int channel = 0; channel < channels_; channel++)
				{
					double sum = 0.0;
					for (int row = 0; row < gen_size_; row++)
					{
						for (int col = 0; col < gen_size_; col++)
						{
							float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
							sum += v;
						}
					}

					//==============if all zero values -> no gradient
					if (fabs(sum) > 1e-6)
					{
						for (int row = 0; row < gen_size_; row++)
						{
							for (int col = 0; col < gen_size_; col++)
							{
								float v = bottom_data[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col];
								bottom_diff[Bid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] += top_diff[Tid + channel * gen_size_ * gen_size_ + row * gen_size_ + col] * (sum - v) / (sum * sum);							
							}
						}
					}
				}
			}		
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelNormalizationResponseV0Layer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelNormalizationResponseV0Layer);
	REGISTER_LAYER_CLASS(DeepHumanModelNormalizationResponseV0);
}
