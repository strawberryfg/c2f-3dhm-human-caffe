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

#include "caffe/h36m.h"

#include<opencv2/opencv.hpp> 
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>


#define black Scalar(0,0,0)
#define yello Scalar(0,255,255)
#define red Scalar(0,0,255)
#define blue Scalar(255, 191, 0)
#define dark_blue Scalar(205, 0, 0)
#define green Scalar(0,255,0)
#define purple Scalar(178,0,178)
#define light_green Scalar(78,255,78)
#define pink Scalar(201, 174, 255)
#define orange Scalar(14, 201, 255)
#define dark_green Scalar(33, 95, 36)
#define sky_blue Scalar(242, 253,34)
#define gray Scalar(134, 153, 147)
#define brown Scalar(0, 128, 128)
#define dark_red Scalar(19, 2, 77)
#define white Scalar(245, 245, 245)
#define middle_blue Scalar(255, 128, 128)
#define middle_green Scalar(123, 223, 64)




namespace caffe {



	//part to all ; all to part
	template <typename Dtype>
	class DeepHumanModelArgmaxHMLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelArgmaxHMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelArgmaxHM"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int gen_size_;
		int channels_;
	};


	//part to all ; all to part
	template <typename Dtype>
	class DeepHumanModelConvert2DLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelConvert2DLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelConvert2D"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int t_dim_;
	};


	//part to all ; all to part
	template <typename Dtype>
	class DeepHumanModelConvert3DLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelConvert3DLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelConvert3D"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int t_dim_;
	};


	//part to all ; all to part
	template <typename Dtype>
	class DeepHumanModelConvertDepthLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelConvertDepthLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelConvertDepth"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int t_dim_;
		int joint_num_;
		double depth_lb_;
		double depth_ub_;
		int root_joint_id_;
	};

	template <typename Dtype>
	class DeepHumanModelGen3DHeatmapInMoreDetailV3Layer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelGen3DHeatmapInMoreDetailV3Layer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelGen3DHeatmapInMoreDetailV3"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int depth_dims_;
		int map_size_;
		int crop_size_;
		int render_sigma_;
		int stride_;
		int joint_num_;
		double x_lower_bound_;
		double x_upper_bound_;
		double y_lower_bound_;
		double y_upper_bound_;
		double z_lower_bound_;
		double z_upper_bound_;
		int output_res_;
	};


	//generated augmented 3d gt
	template <typename Dtype>
	class DeepHumanModelH36MGenAug3DLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelH36MGenAug3DLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelH36MGenAug3D"; }
		virtual inline int ExactNumBottomBlobs() const { return 7; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int joint_num_;
		int crop_bbx_size_;
		string camera_parameters_prefix_;
	private:
		
	};





	//Gen joint 3d blob from xyz heatmap
	//eccv 18 workshop h36m challenge
	template <typename Dtype>
	class DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelH36MChaGenJointFrXYZHeatmapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelH36MChaGenJointFrXYZHeatmap"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
	

		double x_lb_;
		double x_ub_;
		double y_lb_;
		double y_ub_;
		double z_lb_;
		double z_ub_;
		int joint_num_;
	};



	//Generate predicted monocular camera-frame reference based local 3d based on
	//-------1. camera parameters i.e. fx, fy, cx, cy
	//-------2. predicted projection on cropped bounding box (normalized within range [0, 1])
	//-------3. predicted z (global monocular depth; real depth real z)
	template <typename Dtype>
	class DeepHumanModelH36MGenPredMono3DLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelH36MGenPredMono3DLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelH36MGenPredMono3D"; }
		virtual inline int ExactNumBottomBlobs() const { return 7; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int joint_num_;
		string camera_parameters_prefix_;
	};

	template <typename Dtype>
	class DeepHumanModelIntegralXLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelIntegralXLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelIntegralX"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	template <typename Dtype>
	class DeepHumanModelIntegralYLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelIntegralYLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelIntegralY"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	template <typename Dtype>
	class DeepHumanModelIntegralZLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelIntegralZLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelIntegralZ"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	

	template <typename Dtype>
	class DeepHumanModelIntegralVectorLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelIntegralVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelIntegralVector"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double dim_lb_;
		double dim_ub_;
	};


	
	template <typename Dtype>
	class DeepHumanModelNorm3DHMLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelNorm3DHMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelNorm3DHM"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int gen_size_;
		int joint_num_;
		int depth_dims_;
		float hm_threshold_;
	};



	//=======as of Aug 21 2018
	//===slight modifications
	template <typename Dtype>
	class DeepHumanModelNormalizationResponseV0Layer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelNormalizationResponseV0Layer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelNormalizationResponseV0"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int channels_;
		int gen_size_;
                float hm_threshold_;
		
	};


	template <typename Dtype>
	class DeepHumanModelNumericalCoordinateRegressionLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelNumericalCoordinateRegressionLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelNumericalCoordinateRegression"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int S_;
	};



	//output grayscale heatmap (16 joints) to 16 folders
	template <typename Dtype>
	class DeepHumanModelOutputHeatmapSepChannelLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelOutputHeatmapSepChannelLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelOutputHeatmapSepChannel"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 0; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int heatmap_size_;
		int save_size_;
		string save_path_;

		bool output_joint_0_;
		bool output_joint_1_;
		bool output_joint_2_;
		bool output_joint_3_;
		bool output_joint_4_;
		bool output_joint_5_;
		bool output_joint_6_;
		bool output_joint_7_;
		bool output_joint_8_;
		bool output_joint_9_;
		bool output_joint_10_;
		bool output_joint_11_;
		bool output_joint_12_;
		bool output_joint_13_;
		bool output_joint_14_;
		bool output_joint_15_;

		int joint_num_;
	};


	//output skeleton map(with joints visualized on the skeleton map) to file 
	template <typename Dtype>
	class DeepHumanModelOutputJointOnSkeletonMapH36MLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelOutputJointOnSkeletonMapH36MLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelOutputJointOnSkeletonMapH36M"; }
		virtual inline int ExactNumBottomBlobs() const { return 4; }
		virtual inline int ExactNumTopBlobs() const { return 0; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		bool use_raw_rgb_image_;
		bool show_gt_;
		string save_path_;
		int save_size_;
		string image_source_;
		int skeleton_size_;
		bool show_skeleton_;

		float circle_radius_;
		float line_width_;

		bool is_c2f_;
	};


	//part to all ; all to part
	template <typename Dtype>
	class DeepHumanModelSoftmaxHMLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelSoftmaxHMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelSoftmaxHM"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int gen_size_;
		int channels_;
	};


	//softmax on 3d heatmap
	template <typename Dtype>
	class DeepHumanModelSoftmax3DHMLayer : public Layer<Dtype> {
	public:
		explicit DeepHumanModelSoftmax3DHMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHumanModelSoftmax3DHM"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		int gen_size_;
		int joint_num_;
		int depth_dims_;
	};

}  // namespace caffe

#endif  // CAFFE_COMMON_LAYERS_HPP_
