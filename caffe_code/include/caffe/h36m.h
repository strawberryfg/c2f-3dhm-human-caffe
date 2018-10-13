#pragma once
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#define maxlen 1111
using namespace std;
enum basic_settings_h36m
{
	JointNumAll_h36m = 32, BoneNumAll_h36m = 24,
	JointNumPart_h36m = 16, BoneNumPart_h36m = 15,
	TotalVideoName = 840, InfoLines = 7, PairNum_h36m = 120,
	Protocol2TrainNum = 5, ActionNum = 15, h36m_height = 1002, h36m_width = 1000,
};


//coarse to fine annotation train.h5 valid.h5 basic definitions
enum basic_settings_c2f
{
	train_num_c2f = 312188,
	valid_num_c2f = 109867,
	JointNum_c2f = 17,
	BoneNum_c2f = 16
};


enum basic_info_contents
{
	actor_id,
	action_id,
	action_name,
	subaction_id,
	camera_id,
	frame_id,
	video_name

};


enum joint_all_h36m // 32
{
	Hips,             //0
	RightUpLeg,       //1
	RightLeg,         //2
	RightFoot,        //3
	RightToeBase,     //4
	Site_0,           //5
	LeftUpLeg,        //6
	LeftLeg,          //7
	LeftFoot,         //8
	LeftToeBase,      //9
	Site_1,           //10
	Spine,            //11
	Spine1,           //12
	Neck,             //13
	Head,             //14
	Site_2,           //15
	LeftShoulder,     //16
	LeftArm,          //17
	LeftForeArm,      //18
	LeftHand,         //19
	LeftHandThumb,    //20
	Site_3,           //21
	L_Wrist_End,      //22
	Site_4,           //23
	RightShoulder,    //24
	RightArm,         //25
	RightForeArm,     //26
	RightHand,        //27
	RightHandThumb,   //28
	Site_5,           //29
	R_Wrist_End       //30
};

enum joint_part_h36m
{
	part_Hips,         //0
	part_RightUpLeg,   //1
	part_RightLeg,     //2
	part_RightFoot,	   //3
	part_LeftUpLeg,    //4
	part_LeftLeg,      //5
	part_LeftFoot,	   //6
	part_Neck,         //7
	part_Head,         //8
	part_Site_2,	   //9
	part_LeftArm,      //10
	part_LeftForeArm,  //11
	part_LeftHand,	   //12
	part_RightArm,     //13
	part_RightForeArm, //14
	part_RightHand,	   //15
};


enum joint_c2f //coarse to fine 
{
	c2f_part_Hips,         //0
	c2f_part_LeftUpLeg,    //1
	c2f_part_LeftLeg,      //2
	c2f_part_LeftFoot,     //3
	c2f_part_RightUpLeg,   //4
	c2f_part_RightLeg,     //5
	c2f_part_RightFoot,    //6
	c2f_part_Spine1,       //7 (center between thorax and pelvis)
	c2f_part_Neck,         //8
	c2f_part_Head,         //9
	c2f_part_Site_2,       //10
	c2f_part_LeftArm,      //11
	c2f_part_LeftForeArm,  //12
	c2f_part_LeftHand,     //13
	c2f_part_RightArm,     //14
	c2f_part_RightForeArm, //15
	c2f_part_RightHand,    //16
};


const int index_joint_mine_in_c2f[JointNumPart_h36m] =
{
	c2f_part_Hips,         //0
	c2f_part_RightUpLeg,   //1
	c2f_part_RightLeg,     //2
	c2f_part_RightFoot,	  //3
	c2f_part_LeftUpLeg,    //4
	c2f_part_LeftLeg,      //5
	c2f_part_LeftFoot,	  //6
	c2f_part_Neck,         //7
	c2f_part_Head,         //8
	c2f_part_Site_2,	      //9
	c2f_part_LeftArm,      //10
	c2f_part_LeftForeArm,  //11
	c2f_part_LeftHand,	  //12
	c2f_part_RightArm,     //13
	c2f_part_RightForeArm, //14
	c2f_part_RightHand,	  //15
};


const char joint_part_h36m_name[JointNumPart_h36m + 1][maxlen] =
{
	"part_Hips",         //0
	"part_RightUpLeg",   //1
	"part_RightLeg",     //2
	"part_RightFoot",	   //3
	"part_LeftUpLeg",    //4
	"part_LeftLeg",      //5
	"part_LeftFoot",	   //6
	"part_Neck",         //7
	"part_Head",         //8
	"part_Site_2",	   //9
	"part_LeftArm",      //10
	"part_LeftForeArm",  //11
	"part_LeftHand",	   //12
	"part_RightArm",     //13
	"part_RightForeArm", //14
	"part_RightHand",	   //15
	"All"                  //16
};


const bool is_left[JointNumPart_h36m] =
{
	0,     //0
	0,     //1
	0,     //2
	0,     //3
	1,     //4
	1,     //5
	1,     //6
	0,     //7
	0,     //8
	0,     //9
	1,     //10
	1,     //11
	1,     //12
	0,     //13
	0,     //14
	0      //15
};

const int index_joint_in_all[JointNumPart_h36m] =
{
	Hips,          //0  -> 0
	RightUpLeg,    //1  -> 1
	RightLeg,      //2  -> 2
	RightFoot,	   //3  -> 3
	LeftUpLeg,     //4  -> 6
	LeftLeg,       //5  -> 7
	LeftFoot,	   //6  -> 8	
	Neck,          //7  -> 13
	Head,          //8  -> 14
	Site_2,	       //9  -> 15
	LeftArm,       //10 -> 17
	LeftForeArm,   //11 -> 18
	LeftHand,	   //12 -> 19
	RightArm,      //13 -> 25
	RightForeArm,  //14 -> 26
	RightHand,	   //15 -> 27
};

const int index_joint_in_part[JointNumAll_h36m] =
{
	Hips,          //0  -> 0
	RightUpLeg,    //1  -> 1
	RightLeg,      //2  -> 2
	RightFoot,	   //3  -> 3
	-1,            //4  -> -1
	-1,            //5  -> -1
	RightToeBase,  //6  -> 4
	Site_0,        //7  -> 5
	LeftUpLeg,     //8  -> 6
	-1,            //9  -> -1
	-1,            //10 -> -1
	-1,            //11 -> -1
	-1,            //12 -> -1
	LeftLeg,       //13 -> 7	
	LeftFoot,      //14 -> 8
	LeftToeBase,   //15 -> 9
	-1,            //16 -> -1
	Site_1,        //17 -> 10
	Spine,         //18 -> 11
	Spine1,        //19 -> 12
	-1,            //20 -> -1
	-1,            //21 -> -1
	-1,            //22 -> -1
	-1,            //23 -> -1
	-1,            //24 -> -1
	Neck,          //25 -> 13
	Head,          //26 -> 14
	Site_2,        //27 -> 15
};



//======surreal 24 joint definition
const int h36m_all_in_surreal[32] = 
{
	0,                  //0  Hips
	1,                  //1  RightUpLeg
	4,                  //2  RightLeg
	7,                  //3  RightFoot
	-1,                 //4  RightToeBase
	-1,                 //5  Site_0
	2,                  //6  LeftUpLeg
	5,                  //7  LeftLeg
	8,                  //8  LeftFoot
	-1,                 //9  LeftToeBase
	-1,                 //10 Site_1
	-1,                 //11 Spine
	6,                  //12 Spine1
	12,                 //13 Neck
	-1,                 //14 Head
	-1,                 //15 Site_2
	-1,                 //16 LeftShoulder
	14,                 //17 LeftArm
	19,                 //18 LeftForeArm
    21,                 //19 LeftHand
	-1,                 //20 LeftHandThumb
	-1,                 //21 Site_3
	-1,                 //22 L_Wrist_End
	-1,                 //23 Site_4
	-1,                 //24 RightShoulder
	16,                 //25 RightArm
	18,                 //26 RightForeArm
	20,                 //27 RightHand
	-1,                 //28 RightHandThumb
	-1,                 //29 Site_5
	-1,                 //30 R_Wrist_End

};

enum bone_all_h36m //24
{
	bone_Hips_RightUpLeg,            //0
	bone_Hips_LeftUpLeg,             //1
	bone_RightUpLeg_RightLeg,        //2
	bone_RightLeg_RightFoot,         //3
	bone_RightFoot_RightToeBase,     //4
	bone_RightToeBase_Site_0,        //5
	bone_LeftUpLeg_LeftLeg,          //6
	bone_LeftLeg_LeftFoot,           //7
	bone_LeftFoot_LeftToeBase,       //8
	bone_LeftToeBase_Site_1,         //9
	bone_Spine_Spine1,               //10
	bone_Spine1_Neck,                //11
	bone_RightShoulder_RightArm,     //12
	bone_RightArm_RightForeArm,      //13
	bone_RightForeArm_RightHand,     //14
	bone_RightHand_RightHandThumb,   //15
	bone_RightHand_R_Wrist_End,      //16
	bone_LeftShoulder_LeftArm,       //17
	bone_LeftArm_LeftForeArm,        //18
	bone_LeftForeArm_LeftHand,       //19
	bone_LeftHand_LeftHandThumb,     //20
	bone_LeftHand_L_Wrist_End,       //21
	bone_Neck_Head,                  //22
	bone_Head_Site_2	             //23
};

enum bone_part_h36m //16
{
	bone_part_Hips_RightUpLeg,           //0
	bone_part_Hips_LeftUpLeg,            //1
	bone_part_RightUpLeg_RightLeg,       //2
	bone_part_RightLeg_RightFoot,	     //3
	bone_part_LeftUpLeg_LeftLeg,         //4
	bone_part_LeftLeg_LeftFoot,	         //5
	bone_part_Spine_Neck,                //6	
	bone_part_RightShoulder_RightArm,    //7
	bone_part_RightArm_RightForeArm,     //8
	bone_part_RightForeArm_RightHand,	 //9
	bone_part_LeftShoulder_LeftArm,      //10
	bone_part_LeftArm_LeftForeArm,       //11
	bone_part_LeftForeArm_LeftHand,	     //12
	bone_part_Neck_Head,                 //13
	bone_part_Head_Site_2                //14
};

const bool is_left_bone[BoneNumPart_h36m] =
{
	0,    //0
	1,    //1
	0,    //2
	0,    //3
	1,    //4
	1,    //5
	1,    //6
	0,    //7
	0,    //8
	0,    //9
	1,    //10
	1,    //11
	1,    //12
	0,    //13
	0     //14
};

const int asym_bone[BoneNumPart_h36m] = 
{
	bone_part_Hips_LeftUpLeg,
	bone_part_Hips_RightUpLeg,
	bone_part_LeftUpLeg_LeftLeg,
	bone_part_LeftLeg_LeftFoot,
	bone_part_RightUpLeg_RightLeg,
	bone_part_RightLeg_RightFoot,
	bone_part_Spine_Neck,
	bone_part_LeftShoulder_LeftArm,
	bone_part_LeftArm_LeftForeArm,
	bone_part_LeftForeArm_LeftHand,
	bone_part_RightShoulder_RightArm,
	bone_part_RightArm_RightForeArm,
	bone_part_RightForeArm_RightHand,
	bone_part_Neck_Head,
	bone_part_Head_Site_2
};



//-----mostly for challenge 17 joints (1) gen depth label (2) gen 3D hm / marginal hm / hand latent 2.5D hm
const int bones_c2f[BoneNum_c2f][2] =
{
	{ c2f_part_RightUpLeg, c2f_part_Hips },          //0
	{ c2f_part_LeftUpLeg, c2f_part_Hips },           //1
	{ c2f_part_RightLeg, c2f_part_RightUpLeg, },     //2
	{ c2f_part_RightFoot, c2f_part_RightLeg },       //3	
	{ c2f_part_LeftLeg, c2f_part_LeftUpLeg },        //4
	{ c2f_part_LeftFoot, c2f_part_LeftLeg },         //5	
	{ c2f_part_Hips, c2f_part_Spine1 },              //6
	{ c2f_part_Spine1, c2f_part_Neck },              //7
	{ c2f_part_RightArm, c2f_part_Neck },            //8
	{ c2f_part_RightForeArm, c2f_part_RightArm },    //9
	{ c2f_part_RightHand, c2f_part_RightForeArm },   //10
	{ c2f_part_LeftArm, c2f_part_Neck },             //11
	{ c2f_part_LeftForeArm, c2f_part_LeftArm },      //12
	{ c2f_part_LeftHand, c2f_part_LeftForeArm },     //13	
	{ c2f_part_Head, c2f_part_Neck },                //14
	{ c2f_part_Site_2, c2f_part_Head }               //15
};


//h36m part (15)
const int bones_part_h36m_forward_pass[BoneNumPart_h36m][2] =
{
	{ part_Hips, part_LeftUpLeg },        //0
	{ part_LeftUpLeg, part_LeftLeg },     //1
	{ part_LeftLeg, part_LeftFoot },      //2
	{ part_Hips, part_RightUpLeg },       //3
	{ part_RightUpLeg, part_RightLeg },   //4
	{ part_RightLeg, part_RightFoot },    //5
	{ part_Hips, part_Neck },             //6
	{ part_Neck, part_Head },             //7
	{ part_Head, part_Site_2 },           //8
	{ part_Neck, part_LeftArm },          //9
	{ part_LeftArm, part_LeftForeArm },   //10
	{ part_LeftForeArm, part_LeftHand },  //11
	{ part_Neck, part_RightArm },         //12
	{ part_RightArm, part_RightForeArm }, //13
	{ part_RightForeArm, part_RightHand } //14
};


const int bones_part_c2f_forward_pass[BoneNum_c2f][2] =
{
	{ c2f_part_Hips, c2f_part_LeftUpLeg },        //0
	{ c2f_part_LeftUpLeg, c2f_part_LeftLeg },     //1
	{ c2f_part_LeftLeg, c2f_part_LeftFoot },      //2
	{ c2f_part_Hips, c2f_part_RightUpLeg },       //3
	{ c2f_part_RightUpLeg, c2f_part_RightLeg },   //4
	{ c2f_part_RightLeg, c2f_part_RightFoot },    //5
	{ c2f_part_Hips, c2f_part_Neck },             //6
	{ c2f_part_Neck, c2f_part_Head },             //7
	{ c2f_part_Head, c2f_part_Site_2 },           //8
	{ c2f_part_Neck, c2f_part_LeftArm },          //9
	{ c2f_part_LeftArm, c2f_part_LeftForeArm },   //10
	{ c2f_part_LeftForeArm, c2f_part_LeftHand },  //11
	{ c2f_part_Neck, c2f_part_RightArm },         //12
	{ c2f_part_RightArm, c2f_part_RightForeArm }, //13
	{ c2f_part_RightForeArm, c2f_part_RightHand } //14
};




const int bones_all_h36m[BoneNumAll_h36m][2] =
{
	{ RightUpLeg, Hips },          //0
	{ LeftUpLeg, Hips },           //1
	{ RightLeg, RightUpLeg },      //2
	{ RightFoot, RightLeg },       //3
	{ RightToeBase, RightFoot },   //4
	{ Site_0, RightToeBase },      //5
	{ LeftLeg, LeftUpLeg },        //6
	{ LeftFoot, LeftLeg },         //7
	{ LeftToeBase, LeftFoot },     //8
	{ Site_1, LeftToeBase },       //9
	{ Spine, Spine1 },             //10
	{ Spine1, Neck },              //11
	{ RightArm, RightShoulder },   //12	
	{ RightForeArm, RightArm },    //13
	{ RightHand, RightForeArm },   //14
	{ RightHandThumb, RightHand }, //15
	{ R_Wrist_End, RightHand },    //16	
	{ LeftArm, LeftShoulder },     //17
	{ LeftForeArm, LeftArm },      //18
	{ LeftHand, LeftForeArm },     //19
	{ LeftHandThumb, LeftHand },   //20
	{ L_Wrist_End, LeftHand },     //21
	{ Head, Neck },                //22
	{ Site_2, Head }               //23
};

const int bones_part_h36m[BoneNumPart_h36m][2] =
{
	{ RightUpLeg, Hips },          //0
	{ LeftUpLeg, Hips },           //1
	{ RightLeg, RightUpLeg, },      //2
	{ RightFoot, RightLeg },       //3	
	{ LeftLeg, LeftUpLeg },        //4
	{ LeftFoot, LeftLeg },         //5	
	{ Hips, Neck },               //6
	{ RightArm, Neck },   //7
	{ RightForeArm, RightArm },    //8
	{ RightHand, RightForeArm },   //9
	{ LeftArm, Neck },     //10
	{ LeftForeArm, LeftArm },      //11
	{ LeftHand, LeftForeArm },     //12	
	{ Head, Neck },                //13
	{ Site_2, Head }               //14
};



const int color_gt_joint_c2f[JointNum_c2f][3] =
{
	{ 255, 0, 0 },        //0  Hips           Blue
	{ 0, 0, 255 },        //1  LeftUpLeg      Red
	{ 0, 0, 255 },        //2  LeftLeg        Red
	{ 0, 0, 255 },        //3  LeftFoot       Red	
	{ 0, 255, 0 },        //4  RightUpLeg     Green
	{ 0, 255, 0 },        //5  RightLeg       Green
	{ 0, 255, 0 },        //6  RightFoot      Green	
	{ 255, 0, 0 },        //7  Spine1
	{ 255, 0, 0 },        //8  Neck           Blue
	{ 255, 0, 0 },        //9  Head           Blue
	{ 255, 0, 0 },        //10  Site_2         Blue	
	{ 0, 0, 255 },        //11 LeftArm        Red
	{ 0, 0, 255 },        //12 LeftForeArm    Red
	{ 0, 0, 255 },        //13 LeftHand       Red		
	{ 0, 255, 0 },        //14 RightArm       Green
	{ 0, 255, 0 },        //15 RightForeArm   Green
	{ 0, 255, 0 },        //16 RightHand      Green	
};



const int color_gt_joint_all_h36m[JointNumAll_h36m][3] =
{
	{ 255, 0, 0 },        //0  Hips           Blue
	{ 0, 255, 0 },        //1  RightUpLeg     Green
	{ 0, 255, 0 },        //2  RightLeg       Green
	{ 0, 255, 0 },        //3  RightFoot      Green
	{ 0, 255, 0 },        //4  RightToeBase   Green
	{ 0, 255, 0 },        //5  Site_0         Green
	{ 0, 0, 255 },        //6  LeftUpLeg      Red
	{ 0, 0, 255 },        //7  LeftLeg        Red
	{ 0, 0, 255 },        //8  LeftFoot       Red
	{ 0, 0, 255 },        //9  LeftToeBase    Red
	{ 0, 0, 255 },        //10 Site_1         Red
	{ 255, 0, 0 },        //11 Spine          Blue
	{ 255, 0, 0 },        //12 Spine1         Blue
	{ 255, 0, 0 },        //13 Neck           Blue
	{ 255, 0, 0 },        //14 Head           Blue
	{ 255, 0, 0 },        //15 Site           Blue
	{ 0, 0, 255 },        //16 LeftShoulder   Red
	{ 0, 0, 255 },        //17 LeftArm        Red
	{ 0, 0, 255 },        //18 LeftForeArm    Red
	{ 0, 0, 255 },        //19 LeftHand       Red
	{ 0, 0, 255 },        //20 LeftHandThumb  Red
	{ 0, 0, 255 },        //21 Site_3    
	{ 0, 0, 255 },        //22 L_Wrist_End    Red
	{ 0, 0, 255 },        //23 Site_4
	{ 0, 255, 0 },        //24 RightShoulder  Green
	{ 0, 255, 0 },        //25 RightArm       Green
	{ 0, 255, 0 },        //26 RightForeArm   Green
	{ 0, 255, 0 },        //27 RightHand      Green
	{ 0, 255, 0 },        //28 RightHandThumb Green
	{ 0, 255, 0 },        //29 Site_5
	{ 0, 255, 0 },        //30 R_Wrist_End    Green
};

const int color_gt_joint_part_h36m[JointNumPart_h36m][3] =
{
	{ 255, 0, 0 },        //0  Hips           Blue
	{ 0, 255, 0 },        //1  RightUpLeg     Green
	{ 0, 255, 0 },        //2  RightLeg       Green
	{ 0, 255, 0 },        //3  RightFoot      Green	
	{ 0, 0, 255 },        //4  LeftUpLeg      Red
	{ 0, 0, 255 },        //5  LeftLeg        Red
	{ 0, 0, 255 },        //6  LeftFoot       Red			
	{ 255, 0, 0 },        //7  Neck           Blue
	{ 255, 0, 0 },        //8  Head           Blue
	{ 255, 0, 0 },        //9  Site_2         Blue	
	{ 0, 0, 255 },        //10 LeftArm        Red
	{ 0, 0, 255 },        //11 LeftForeArm    Red
	{ 0, 0, 255 },        //12 LeftHand       Red		
	{ 0, 255, 0 },        //13 RightArm       Green
	{ 0, 255, 0 },        //14 RightForeArm   Green
	{ 0, 255, 0 },        //15 RightHand      Green	
};



const int color_pred_bone_all_h36m[BoneNumAll_h36m][3] =
{
	{ 255, 0, 0 },        //0   Hips          ->   RightUpLeg
	{ 255, 0, 0 },        //1   Hips          ->   LeftUpLeg
	{ 0, 255, 0 },        //2   RightUpLeg    ->   RightLeg
	{ 0, 255, 0 },        //3   RightLeg      ->   RightFoot
	{ 0, 255, 0 },        //4   RightFoot     ->   RightToeBase
	{ 0, 255, 0 },        //5   RightToeBase  ->   Site_0
	{ 0, 0, 255 },        //6   LeftUpLeg     ->   LeftLeg
	{ 0, 0, 255 },        //7   LeftLeg       ->   LeftFoot
	{ 0, 0, 255 },        //8   LeftFoot      ->   LeftToeBase
	{ 0, 0, 255 },        //9   LeftToeBase   ->   Site_1
	{ 255, 0, 0 },        //10  Spine         ->   Spine1
	{ 255, 0, 0 },        //11  Spine1        ->   Neck
	{ 0, 255, 0 },        //12  RightShoulder ->   RightArm
	{ 0, 255, 0 },        //13  RightArm      ->   RightForeArm
	{ 0, 255, 0 },        //14  RightForeArm  ->   RightHand
	{ 0, 255, 0 },        //15  RightHand     ->   RightHandThumb
	{ 0, 255, 0 },        //16  RightHand     ->   R_Wrist_End	
	{ 0, 0, 255 },        //17  LeftShoulder  ->   LeftArm
	{ 0, 0, 255 },        //18  LeftArm       ->   LeftForeArm
	{ 0, 0, 255 },        //19  LeftForeArm   ->   LeftHand
	{ 0, 0, 255 },        //20  LeftHand      ->   LeftHandThumb
	{ 0, 0, 255 },        //21  LeftHand      ->   L_Wrist_End	
	{ 255, 0, 0 },        //22  Neck          ->   Head	
	{ 255, 0, 0 }         //23  Head          ->   Site_2
};

const int color_gt_bone_all_h36m[BoneNumAll_h36m][3] =
{
	{ 0, 0, 236 },        //0   Hips          ->   RightUpLeg
	{ 0, 0, 236 },        //1   Hips          ->   LeftUpLeg
	{ 0, 0, 236 },        //2   RightUpLeg    ->   RightLeg
	{ 0, 0, 236 },        //3   RightLeg      ->   RightFoot
	{ 0, 0, 236 },        //4   RightFoot     ->   RightToeBase
	{ 0, 0, 236 },        //5   RightToeBase  ->   Site_0
	{ 0, 0, 236 },        //6   LeftUpLeg     ->   LeftLeg
	{ 0, 0, 236 },        //7   LeftLeg       ->   LeftFoot
	{ 0, 0, 236 },        //8   LeftFoot      ->   LeftToeBase
	{ 0, 0, 236 },        //9   LeftToeBase   ->   Site_1
	{ 0, 0, 236 },        //10  Spine         ->   Spine1
	{ 0, 0, 236 },        //11  Spine1        ->   Neck
	{ 0, 0, 236 },        //12  RightShoulder ->   RightArm
	{ 0, 0, 236 },        //13  RightArm      ->   RightForeArm
	{ 0, 0, 236 },        //14  RightForeArm  ->   RightHand
	{ 0, 0, 236 },        //15  RightHand     ->   RightHandThumb
	{ 0, 0, 236 },        //16  RightHand     ->   R_Wrist_End	
	{ 0, 0, 236 },        //17  LeftShoulder  ->   LeftArm
	{ 0, 0, 236 },        //18  LeftArm       ->   LeftForeArm
	{ 0, 0, 236 },        //19  LeftForeArm   ->   LeftHand
	{ 0, 0, 236 },        //20  LeftHand      ->   LeftHandThumb
	{ 0, 0, 236 },        //21  LeftHand      ->   L_Wrist_End	
	{ 0, 0, 236 },        //22  Neck          ->   Head	
	{ 0, 0, 236 }         //23  Head          ->   Site_2
};


const int color_pred_bone_part_h36m[BoneNumPart_h36m][3] =
{
	{ 255, 0, 0 },        //0   Hips          ->   RightUpLeg
	{ 255, 0, 0 },        //1   Hips          ->   LeftUpLeg
	{ 0, 255, 0 },        //2   RightUpLeg    ->   RightLeg
	{ 0, 255, 0 },        //3   RightLeg      ->   RightFoot	
	{ 0, 0, 255 },        //4   LeftUpLeg     ->   LeftLeg
	{ 0, 0, 255 },        //5   LeftLeg       ->   LeftFoot	
	{ 255, 0, 0 },        //6   Spine         ->   Neck
	{ 0, 255, 0 },        //7   RightShoulder ->   RightArm
	{ 0, 255, 0 },        //8   RightArm      ->   RightForeArm
	{ 0, 255, 0 },        //9  RightForeArm  ->   RightHand	
	{ 0, 0, 255 },        //10  LeftShoulder  ->   LeftArm
	{ 0, 0, 255 },        //11  LeftArm       ->   LeftForeArm
	{ 0, 0, 255 },        //12  LeftForeArm   ->   LeftHand	
	{ 255, 0, 0 },        //13  Neck          ->   Head	
	{ 255, 0, 0 }         //14  Head          ->   Site_2
};




const int color_pred_bone_c2f[BoneNum_c2f][3] =
{
	{ 255, 0, 0 },        //0   Hips          ->   RightUpLeg
	{ 255, 0, 0 },        //1   Hips          ->   LeftUpLeg
	{ 0, 255, 0 },        //2   RightUpLeg    ->   RightLeg
	{ 0, 255, 0 },        //3   RightLeg      ->   RightFoot	
	{ 0, 0, 255 },        //4   LeftUpLeg     ->   LeftLeg
	{ 0, 0, 255 },        //5   LeftLeg       ->   LeftFoot	
	{ 255, 0, 0 },        //6   Spine1        ->   Hips
	{ 255, 0, 0 },        //7   Spine1        ->   Neck
	{ 0, 255, 0 },        //8   RightShoulder ->   RightArm
	{ 0, 255, 0 },        //9   RightArm      ->   RightForeArm
	{ 0, 255, 0 },        //10  RightForeArm  ->   RightHand	
	{ 0, 0, 255 },        //11  LeftShoulder  ->   LeftArm
	{ 0, 0, 255 },        //12  LeftArm       ->   LeftForeArm
	{ 0, 0, 255 },        //13  LeftForeArm   ->   LeftHand	
	{ 255, 0, 0 },        //14  Neck          ->   Head	
	{ 255, 0, 0 }         //15  Head          ->   Site_2
};


const int color_gt_bone_part_h36m[BoneNumPart_h36m][3] =
{
	{ 0, 0, 236 },        //0   Hips          ->   RightUpLeg
	{ 0, 0, 236 },        //1   Hips          ->   LeftUpLeg
	{ 0, 0, 236 },        //2   RightUpLeg    ->   RightLeg
	{ 0, 0, 236 },        //3   RightLeg      ->   RightFoot	
	{ 0, 0, 236 },        //4   LeftUpLeg     ->   LeftLeg
	{ 0, 0, 236 },        //5   LeftLeg       ->   LeftFoot	
	{ 0, 0, 236 },        //6   Spine         ->   Neck
	{ 0, 0, 236 },        //7   RightShoulder ->   RightArm
	{ 0, 0, 236 },        //8   RightArm      ->   RightForeArm
	{ 0, 0, 236 },        //9  RightForeArm  ->   RightHand	
	{ 0, 0, 236 },        //10  LeftShoulder  ->   LeftArm
	{ 0, 0, 236 },        //11  LeftArm       ->   LeftForeArm
	{ 0, 0, 236 },        //12  LeftForeArm   ->   LeftHand	
	{ 0, 0, 236 },        //13  Neck          ->   Head	
	{ 0, 0, 236 }         //14  Head          ->   Site_2
};



const int skeleton_color_gt_bone_all_h36m[BoneNumAll_h36m][3] =
{
	{ 186, 118, 186 },        //0   Hips          ->   RightUpLeg
	{ 215, 194, 255 },        //1   Hips          ->   LeftUpLeg
	{ 120, 196, 89 },         //2   RightUpLeg    ->   RightLeg
	{ 52, 134, 254 },         //3   RightLeg      ->   RightFoot
	{ 63, 245, 255 },         //4   RightFoot     ->   RightToeBase
	{ 238, 185, 63 },         //5   RightToeBase  ->   Site_0
	{ 129, 155, 202 },        //6   LeftUpLeg     ->   LeftLeg
	{ 193, 152, 120 },        //7   LeftLeg       ->   LeftFoot
	{ 198, 198, 198 },        //8   LeftFoot      ->   LeftToeBase
	{ 63, 159, 159 },         //9   LeftToeBase   ->   Site_1
	{ 63, 63, 159 },          //10  Spine         ->   Spine1
	{ 159, 159, 111 },        //11  Spine1        ->   Neck
	{ 63, 63, 255 },          //12  RightShoulder ->   RightArm
	{ 161, 213, 181 },        //13  RightArm      ->   RightForeArm
	{ 138, 229, 220 },        //14  RightForeArm  ->   RightHand
	{ 202, 246, 119 },        //15  RightHand     ->   RightHandThumb
	{ 189, 129, 237 },        //16  RightHand     ->   R_Wrist_End	
	{ 178, 185, 188 },        //17  LeftShoulder  ->   LeftArm
	{ 162, 138, 229 },        //18  LeftArm       ->   LeftForeArm
	{ 115, 241, 251 },        //19  LeftForeArm   ->   LeftHand
	{ 188, 237, 249 },        //20  LeftHand      ->   LeftHandThumb
	{ 159, 111, 63 },         //21  LeftHand      ->   L_Wrist_End	
	{ 227, 220, 209 },        //22  Neck          ->   Head	
	{ 141, 113, 154 }         //23  Head          ->   Site_2
};

const int skeleton_color_gt_bone_part_h36m[BoneNumPart_h36m + 1][3] =
{
	{ 178, 0, 178 },            //0  Hips          ->   RightUpLeg
	{ 33, 95, 36 },             //1  Hips          ->   LeftUpLeg
	{ 205, 0, 0 },              //2  RightUpLeg    ->   RightLeg
	{ 201, 174, 255 },          //3  RightLeg      ->   RightFoot
	{ 0, 0, 255 },              //4  LeftUpLeg     ->   LeftLeg
	{ 14, 201, 255 },           //5  LeftLeg       ->   LeftFoot	
	{ 78, 255, 78 },            //6  Spine         ->   Neck
	{ 134, 153, 147 },          //7  RightShoulder ->   RightArm
	{ 19, 2, 77 },              //8  RightArm      ->   RightForeArm
	{ 0, 128, 128 },            //9  RightForeArm  ->   RightHand	
	{ 245, 245, 245 },          //10 LeftShoulder  ->   LeftArm
	{ 255, 128, 128 },          //11 LeftArm       ->   LeftForeArm
	{ 123, 223, 64 },           //12 LeftForeArm   ->   LeftHand
	{ 242, 253, 34 },           //13 Neck          ->   Head	
	{ 0, 255, 255 },            //14 Head          ->   Site_2
	{0, 0, 0, }                 // the last one: background
};


//as of Jul 31 2018 (next: extend to mpii)
const int skeleton_color_gt_bone_part_h36m_v0[BoneNumPart_h36m + 1][3]=
{
	{ 151, 87, 121 },            //0  Hips          ->   RightUpLeg
	{ 167, 231, 197 },             //1  Hips          ->   LeftUpLeg
	{ 85, 152, 188 },              //2  RightUpLeg    ->   RightLeg
	{ 199, 153, 96 },          //3  RightLeg      ->   RightFoot
	{ 191, 141, 113 },              //4  LeftUpLeg     ->   LeftLeg
	{ 119, 165, 232 },           //5  LeftLeg       ->   LeftFoot	
	{ 140, 140, 140 },            //6  Spine         ->   Neck
	{ 234, 218, 188 },          //7  RightShoulder ->   RightArm
	{ 140, 139, 66 },              //8  RightArm      ->   RightForeArm
	{ 84, 83, 232 },            //9  RightForeArm  ->   RightHand	
	{ 84, 100, 130 },          //10 LeftShoulder  ->   LeftArm
	{ 178, 179, 252 },          //11 LeftArm       ->   LeftForeArm
	{ 234, 235, 84 },           //12 LeftForeArm   ->   LeftHand
	{ 65, 135, 226 },           //13 Neck          ->   Head	
	{ 253, 183, 92 },            //14 Head          ->   Site_2
	{ 0, 0, 0, }                 // the last one: background
};

const int subject_id_train_protocol2[Protocol2TrainNum] = 
{
	1, 5, 6, 7, 8
};

const char action_names[ActionNum][maxlen] = 
{
	"Directions",
	"Discussion", 
	"Eating", 
	"Greeting", 
	"Phoning", 
	"Posing", 
	"Purchases", 
	"Sitting", 
	"SittingDown", 
	"Smoking", 
	"TakingPhoto", 
	"Waiting", 
	"Walking", 
	"WalkingDog", 
	"WalkingTogether"
};



//preserves the depth in the hierarchical tree level human3.6m
const int level[JointNumPart_h36m] =
{
	1,                 //part_Hips
	2,                 //part_RightUpLeg
	3,                 //part_RightLeg
	4,                 //part_RightFoot
	2,                 //part_LeftUpLeg
	3,                 //part_LeftLeg
	4,                 //part_LeftFoot
	0,                 //part_Neck
	1,                 //part_Head
	2,                 //part_Site_2
	1,                 //part_LeftArm
	2,                 //part_LeftForeArm
	3,                 //part_LeftHand
	1,                 //part_RightArm
	2,                 //part_RightForeArm
	3                  //part_RightHand
};


//use camera parameters read from the file to generate global 2d projection
double *ProjectPointRadial(double *joint_3d, int index, char *camera_prefix, bool project_partial_joint);
