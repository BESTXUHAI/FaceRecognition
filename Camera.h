#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include "Facenet.h"

using namespace std;


/// <summary>
///Camera class for displaying video images
/// </summary>
class Camera
{
public:
	Camera(string videopath, int mode);
	void DramRect(cv::Mat& img, vector<Face>& faces, vector<string>& label);
	void videoShow(MTCNNDetector* detector, Facenet* facenet);
	void faceRecognition(cv::Mat& img, vector<Face>& faces, vector<string>& label, MTCNNDetector* detector, Facenet* facenet);
	~Camera();
private:
	string videoPath_;
	//Display mode 0: camera, 1: video, 2: picture
	int mode_;
	
	cv::VideoCapture cap_;


};

