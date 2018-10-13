#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_human_model_layers.hpp"


//====input: aug 2d / gt 3d (for gt z)
//====output: aug 3d
namespace caffe {

	template <typename Dtype>
	void DeepHumanModelH36MGenAug3DLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		joint_num_ = this->layer_param_.gen_aug_3d_param().joint_num();
		camera_parameters_prefix_ = this->layer_param_.gen_aug_3d_param().camera_parameters_prefix();
		crop_bbx_size_ = this->layer_param_.gen_aug_3d_param().crop_bbx_size();
		//=====because the augmentation layer generate raw coordinate in cropped bbx within e.g. range [0, 256]

	}
	template <typename Dtype>
	void DeepHumanModelH36MGenAug3DLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_ * 3);

		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHumanModelH36MGenAug3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* aug_2d_data = bottom[0]->cpu_data();     //augmented 2d in cropped bounding box [0, 256]
		const Dtype* gt_mono_3d_data = bottom[1]->cpu_data();     //gt 3d(in camera frame)
		const Dtype* bbx_x1_data = bottom[2]->cpu_data();     //bbx_x1
		const Dtype* bbx_y1_data = bottom[3]->cpu_data();     //bbx_y1
		const Dtype* bbx_x2_data = bottom[4]->cpu_data();     //bbx_x2
		const Dtype* bbx_y2_data = bottom[5]->cpu_data();     //bbx_y1
		const Dtype* index_data = bottom[6]->cpu_data();      //image index to index camera intrinsic parameters
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int Pid = t * joint_num_ * 2, Gid = t * joint_num_ * 3;
			int Tid = t * joint_num_ * 3;

			//================read camera parameters (here only use intrinsic fx fy u0 v0)

			int index = (int)index_data[t];

			//Read camera parameters
			char camera_file[maxlen];
			sprintf(camera_file, "%s%d%s", camera_parameters_prefix_.c_str(), index, ".txt");
			FILE *fin_camera = fopen(camera_file, "r");

			//======camera intrinsic parameter array
			double R[3][3], T[3], f[2], c[2], K[3], p[2], P[3];
			double arr[3], X[3], XX[2], r2, radial, tang, tmp[2], add_a[2], add_b[2], XXX[2], ans_a[2], ans_b[2];

			char ts[maxlen];
			fscanf(fin_camera, "%s", ts);
			//R
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					fscanf(fin_camera, "%lf", &R[i][j]);
				}
			}

			//T
			fscanf(fin_camera, "%s", ts);
			for (int i = 0; i < 3; i++)
			{
				double t;
				fscanf(fin_camera, "%lf", &t);
				T[i] = t;
			}


			//f
			fscanf(fin_camera, "%s", ts);
			for (int i = 0; i < 2; i++)
				fscanf(fin_camera, "%lf", &f[i]);

			//c
			fscanf(fin_camera, "%s", ts);
			for (int i = 0; i < 2; i++)
			{
				double t;
				fscanf(fin_camera, "%lf", &t);
				c[i] = t;
			}

			//k
			fscanf(fin_camera, "%s", ts);
			for (int i = 0; i < 3; i++)
			{
				double t;
				fscanf(fin_camera, "%lf", &t);
				K[i] = t;
			}

			//p
			fscanf(fin_camera, "%s", ts);
			for (int i = 0; i < 2; i++)
			{
				fscanf(fin_camera, "%lf", &p[i]);
			}


			fclose(fin_camera);
			//================End of reading camera parameters

			//================reproject2Dto3D
			for (int i = 0; i < joint_num_; i++)
			{
				double proj_u = aug_2d_data[Pid + i * 2], proj_v = aug_2d_data[Pid + i * 2 + 1]; // in the range of [0, 1]
				double bbx_x1 = bbx_x1_data[t]; 
				double bbx_y1 = bbx_y1_data[t];
				double bbx_x2 = bbx_x2_data[t];
				double bbx_y2 = bbx_y2_data[t];
				//=====now that proj_u, proj_v are in range[0, 256] [0, crop_bbx_size_]
				proj_u /= double(crop_bbx_size_);
				proj_v /= double(crop_bbx_size_);
				//=====get projection on raw 1000x1002 image
				proj_u = proj_u * (bbx_x2 - bbx_x1) + bbx_x1;
				proj_v = proj_v * (bbx_y2 - bbx_y1) + bbx_y1; 
				int Gid = t * joint_num_ * 3;
				double gt_z = gt_mono_3d_data[Gid + i * 3 + 2];
				double aug_x = (proj_u - c[0]) / f[0] * gt_z;
				double aug_y = (proj_v - c[1]) / f[1] * gt_z;
				top_data[Tid + i * 3] = aug_x;
				top_data[Tid + i * 3 + 1] = aug_y;
				top_data[Tid + i * 3 + 2] = gt_z;
			}
		}
	}

	template <typename Dtype>
	void DeepHumanModelH36MGenAug3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHumanModelH36MGenAug3DLayer);
#endif

	INSTANTIATE_CLASS(DeepHumanModelH36MGenAug3DLayer);
	REGISTER_LAYER_CLASS(DeepHumanModelH36MGenAug3D);
}