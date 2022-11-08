#include "Camera.h"
Camera::Camera(string videopath, int mode)
{
	videoPath_ = videopath;
	
	mode_ = mode;
	if (videopath.find(".mp4") != -1 && mode==1)
		cap_ = cv::VideoCapture(videopath);
	else if(mode == 0)
		cap_ = cv::VideoCapture(0);

}
//Draw face rectangle and display face label
void Camera::DramRect(cv::Mat& img, vector<Face>& faces, vector<string>& label)
{
	for (int i = 0; i < faces.size(); i++)
	{	
	
		//Get Rectangular Box
		cv::Rect rect = faces[i].bbox.getSquare().getRect();
		cv::Rect screct(int(rect.tl().x), int(rect.tl().y ), 
			int(rect.width), int(rect.height));
		
		cv::Scalar color;
		
		if (label[i] != "none")color = cv::Scalar(0, 255, 0);
		else color = cv::Scalar(0, 0, 255);
		
		cv::rectangle(img, screct, color);
		double frntsize = screct.width / 100.0;
		//Show Label
		cv::putText(img, label[i], cv::Point(screct.tl().x, screct.tl().y),
			cv::FONT_HERSHEY_TRIPLEX, frntsize, color);
	}
}

void Camera::faceRecognition(cv::Mat& img, vector<Face>& faces, vector<string>& label, MTCNNDetector* detector, Facenet* facenet)
{
	double threshold = 1.1;
	for (int i = 0; i < faces.size(); i++)
	{
		//Align each face
		vector<Face> def{ faces[i] };
		detector->faceAlign(img, def, "test/detected/dt");
		cv::Mat alignedImg = cv::imread("test/detected/dt_align0.jpg");
		//Extract features
		cv::Mat feat = facenet->featureExtract(alignedImg);
		//faceRecognition
		label[i] = facenet->faceRecognition(feat, threshold);

	}
}


//Read videos or pictures
void Camera::videoShow(MTCNNDetector* detector, Facenet* facenet)
{
	
	//Try opening the video
	if (mode_ != 2 && !cap_.isOpened())
	{
		return;
	}
	cv::Mat img;
	
	if (mode_ == 0 || mode_ == 1)
	{
		int frame = 0;
		vector<Face> faces;
		vector<string> label;
		int lastface_num = 0;
		while (cap_.read(img))
		{
			int h = img.rows;
			int w = img.cols;
			float scale = 1.0;
			cv::Mat scimg(img);
			if (h > 108)
			{
				scale = h / 108.0;
				h = h / scale;
				w = w / scale;
				cv::resize(img, scimg, cv::Size(w, h), 0, 0, 3);
			}
			frame++;
			//In order to monitor in real time and reduce the size, 5 frames are detected at a time
			
			if (frame % 5 == 0)
			{
				//Parameters: image to be detected, minimum face, maximum face proportion, image pyramid scaling
				faces = detector->detect(scimg, 20.f, 0.5f,0.709f);
				
				for (int i = 0; i < faces.size(); i++)
				{
					//Enlarge detection box
					faces[i].faceScale(scale);
				}
				if(label.size()<faces.size())label.resize(faces.size(), "none");
				//Face recognition once per 10 frames
				if (frame % 2 ==0)
				{
					faceRecognition(img, faces, label, detector, facenet);
					lastface_num = faces.size();
				}
			}
			
			
			DramRect(img, faces, label);
			
			
			cv::imshow("video", img);
			//Can be adjusted appropriately
			int key = cv::waitKey(25);
			if (key == 'q')
				break;
			
		}
		cap_.release();
		
	}
	else if (mode_ == 2)
	{
		img = cv::imread(videoPath_);
		int h = img.rows;
		int w = img.cols;
		if (h > 480)
		{
			float scale = h / 480.0;
			h = h / scale;
			w = w / scale;
			cv::resize(img, img, cv::Size(w, h));
		}
		
		vector<Face> faces = detector->detect(img, 20.f, 1.f,0.709f);
		vector<string> label(faces.size());
		label.resize(faces.size(),"none");

		faceRecognition(img, faces, label, detector, facenet);
	
		DramRect(img, faces, label);

		cv::imshow("img", img);
		int key = cv::waitKey(10000);
		if (key == 'q')
			return;
	}
	
}



Camera::~Camera() 
{
	if (cap_.isOpened())
		cap_.release();
}