#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "random"
#include "ctime"
using namespace cv;

namespace caffe {


    //======adaptive weight euclidean L2 loss
    template <typename Dtype>
    class AdaptiveWeightEucLossLayer : public LossLayer<Dtype> {
    public:
        explicit AdaptiveWeightEucLossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "AdaptiveWeightEucLoss"; }
        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int ExactNumBottomBlobs() const { return -1; } //====prevent it from checking bottom blob counts
        
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


        int num_of_losses_;
        double avg_abs_diff_each_dim[111]; //===average of |x_i - x_i ^GT| (each dimension of each objective func) in the batch 
    };

   //Add vector by constant
    template <typename Dtype>
    class AddVectorByConstantLayer : public Layer<Dtype> {
    public:
        explicit AddVectorByConstantLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "AddVectorByConstant"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        float add_value_;
        int dim_size_;
    };




    //one  (one element value predicted by CNN) + the whole vector
    template <typename Dtype>
    class AddVectorBySingleVectorLayer : public Layer<Dtype> {
    public:
        explicit AddVectorBySingleVectorLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "AddVectorBySingleVector"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        int dim_size_;
    };


    //Cross Validation ten-fold leave one out choose a index from several indexes
    template <typename Dtype>
    class CrossValidationRandomChooseIndexLayer : public Layer<Dtype> {
    public:
        explicit CrossValidationRandomChooseIndexLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "CrossValidationRandomChooseIndex"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        
    };

    //Multiply the [-1, 1] value by standard deviation and add the mean average number
    template <typename Dtype>
    

	//universal heatmap render (gaussian heatmap) layer
	template <typename Dtype>
	class GenHeatmapAllChannelsLayer : public Layer<Dtype> {
	public:
		explicit GenHeatmapAllChannelsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GenHeatmapAllChannels"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; } 
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int gen_size_;
		double render_sigma_;
		int joint_num_;
		bool all_one_; //binary classification (G-RMI)

        bool use_cpm_render_;
        bool use_baseline_render_;
        int crop_size_;
        int stride_;
        int grid_size_;
	};



 //Generate random index
    template <typename Dtype>
    class GenRandIndexLayer : public Layer<Dtype> {
    public:
        explicit GenRandIndexLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "GenRandIndex"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        int index_lower_bound_;
        int index_upper_bound_;
        int batch_size_;
        string missing_index_file_;
        int missing_index_[11111];
        int num_of_missing_;

        int rand_generator_option_;
    };

    //Generate sequential index
    template <typename Dtype>
    class GenSequentialIndexLayer : public Layer<Dtype> {
    public:
        explicit GenSequentialIndexLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "GenSequentialIndex"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string current_index_file_path_; //stores only one single value denoting the current index        
        int batch_size_;
        int num_of_samples_;
        int start_index_;
    };




    //Generate unified online data and label w/o lmdb
    //-------1. image_index (to index file containing image path)
    //-------2. center_x (on raw image)
    //-------3. center_y 
    //-------4. scale_provided (real scale / a constant which is mostly 200.0
    //-------5. gt_joint_2d_raw 
    template <typename Dtype>
    class GenUnifiedDataAndLabelLayer : public Layer<Dtype> {
    public:
        explicit GenUnifiedDataAndLabelLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "GenUnifiedDataAndLabel"; }
        virtual inline int ExactNumBottomBlobs() const { return 5; }
        virtual inline int ExactNumTopBlobs() const { return 2; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        TransformationParameter param_;
        int crop_x_, crop_y_;
        int joint_num_;
        string file_name_file_prefix_;
        double minus_pixel_value_;
    private:
        void augmentation_scale(Mat& img_src, Mat& img_temp, Dtype *joint_data, Dtype scale_self, Dtype *objpos_x, Dtype *objpos_y);
        void RotatePoint(cv::Point2f& p, Mat R);
        void augmentation_rotate(Mat& img_src, Mat& img_dst, Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y);
        bool onPlane(cv::Point p, Size img_size);
        void augmentation_croppad(Mat& img_src, Mat& img_dst, Dtype *joint_data, Dtype *objpos_x, Dtype *objpos_y);
        void swapLeftRight(Dtype *joint_data);
        void augmentation_flip(Mat& img_src, Mat& img_aug, Dtype *joint_data, Dtype *objpos_x);

    };


    //square root 3D Joint Location Loss
    template <typename Dtype>
    class Joint3DSquareRootLossLayer : public LossLayer<Dtype> {
    public:
        explicit Joint3DSquareRootLossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Joint3DSquareRootLoss"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


        int joint_num_;
    };



	//Jenson Shannon Regularization
	template <typename Dtype>
	class JSRegularizationLossLayer : public LossLayer<Dtype> {
	public:
		explicit JSRegularizationLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "JSRegularizationLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double min_eps_;
		int channel_num_;
	};


    //Write the blob to disk
    template <typename Dtype>
    class OutputBlobLayer : public Layer<Dtype> {
    public:
        explicit OutputBlobLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "OutputBlob"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 0; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string save_path_;
        string blob_name_;

		bool if_per_section_output_;
		int per_section_row_num_;
		int per_section_col_num_;
    };
	
	//Output one-channel heat map to disk     
	template <typename Dtype>
	class OutputHeatmapOneChannelLayer : public Layer<Dtype> {
	public:
		explicit OutputHeatmapOneChannelLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "OutputHeatmapOneChannel"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 0; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string save_path_;
		int save_size_;

		int heatmap_size_;
	};



    //Read blob from disk file just one file
    template <typename Dtype>
    class ReadBlobFromFileLayer : public Layer<Dtype> {
    public:
        explicit ReadBlobFromFileLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ReadBlobFromFile"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string file_path_;
        int num_to_read_;
        double t_data[11111];
        int batch_size_;
    };




    //Multiplies the RGB by a factor
    template <typename Dtype>
    class MulRGBLayer : public Layer<Dtype> {
    public:
        explicit MulRGBLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "MulRGB"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        float mul_factor_;
    };




    //Read blob from disk file indexing
    template <typename Dtype>
    class ReadBlobFromFileIndexingLayer : public Layer<Dtype> {
    public:
        explicit ReadBlobFromFileIndexingLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ReadBlobFromFileIndexing"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string file_prefix_;
        int num_to_read_;
        double t_data[11111];
    };



	//read image from file name
	template <typename Dtype>
	class ReadImageFromFileNameLayer : public Layer<Dtype> {
	public:
		explicit ReadImageFromFileNameLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ReadImageFromFileName"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		bool pad_square_; //whether to pad width or height to maintain the same size in either side
		int resize_size_;
		int channel_num_; //1: gray 3: RGB
		string file_name_file_prefix_; //"file name" file prefix 
		bool pad_to_a_constant_size_before_resize_;
		int pad_to_constant_size_;
	};


    //Read image from "image path file" (one file)
    template <typename Dtype>
    class ReadImageFromImagePathFileLayer : public Layer<Dtype> {
    public:
        explicit ReadImageFromImagePathFileLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ReadImageFromImagePathFile"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string image_path_file_path_;
        string current_index_file_path_; //stores only one single value denoting the current index
        int batch_size_;
        int num_of_samples_;
        int resize_image_size_;
    };


//read image from file
    template <typename Dtype>
    class ReadImageLayer : public Layer<Dtype> {
    public:
        explicit ReadImageLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ReadImage"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string read_path_;
        int resize_size_;
        int zero_pad_; //%08d for example

        string image_suffix_;
    };

    //Read index from disk file just one file (for testing)
    template <typename Dtype>
    class ReadIndexFromFileLayer : public Layer<Dtype> {
    public:
        explicit ReadIndexFromFileLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ReadIndexFromFile"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string index_file_path_;
        string current_index_file_path_; //stores only one single value denoting the current index
        int batch_size_;
        int num_of_samples_;
    };
    


    //Multiply the vector by a constant number
    template <typename Dtype>
    class ScaleVectorLayer : public Layer<Dtype> {
    public:
        explicit ScaleVectorLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ScaleVector"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        float scale_factor_;
        int dim_size_;
    };

}

#endif  // CAFFE_COMMON_LAYERS_HPP_
