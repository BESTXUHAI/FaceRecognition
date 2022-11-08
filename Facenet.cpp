#include "Facenet.h"
#include "utils.h"
Facenet::Facenet(string modelpb, string modeltxt)
{
	_net = cv::dnn::readNetFromTensorflow(modelpb);
	if (_net.empty()) {
		cout << "Failed to load facenet" << endl;
		throw std::invalid_argument("facenet loading error");
	}
	
}

void Facenet::datasetExtract(string dataset, string aligned)
{
	string labelpath = dataset + "/label.txt";
	//Get Label
	std::ifstream file;
	file.open(labelpath, std::ios::in);
	if (!file.is_open()) {
		std::cout << "open file failed!" << std::endl;
	}
	//Reading files
	std::string str;
	while (std::getline(file, str)) {
		int i = str.find("-");
		if (i == -1)continue;
		_nametolabel[str.substr(0, i)] = str.substr(i + 1);
		
	}
	file.close();

	//Extract aligned image features
	_feat.clear();
	_vimgname.clear();
	vector<string> imgspath;
	cv::glob(aligned + "/*.jpg", imgspath);
	for (int i = 0; i < imgspath.size(); i++)
	{
		int st = 0;
		for (int j = 0; j < imgspath[i].size(); j++)
		{
			if (imgspath[i][j] == '\\')imgspath[i][j] = '/';
			if (imgspath[i][j] == '/')st = j + 1;
		}
		int ed = imgspath[i].find("_align");
		int len = ed - st;
		if (len <= 0)continue;
		
		cv::Mat img = cv::imread(imgspath[i]);
		if (img.empty())continue;
		cv::Mat feat = featureExtract(img);
		_feat.push_back(feat);
		
		_vimgname.push_back(imgspath[i].substr(st, len) + ".jpg");
	}

}


cv::Mat Facenet::featureExtract(const cv::Mat& img)
{
	
	//Convert the image to floating point and normalize it
	cv::Mat fimg = convertImg(img);

	//img = imgStandardization(img);
	//cv::Mat inputBlob =	cv::dnn::blobFromImage(img, 1.0, cv::Size(160,160));
	cv::Mat inputBlob = cv::dnn::blobFromImage(fimg, IMG_INV_STDDEV, cv::Size(160, 160),
		cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

	//Set Model Input
	_net.setInput(inputBlob);
	//Forward propagation
	vector<cv::Mat> outputBlobs;
	_net.forward(outputBlobs);
	return outputBlobs[0].clone();
}

double Facenet::getSimilarity(const cv::Mat& feat1, const cv::Mat& feat2)
{
	//European distance
	return cv::norm(feat1 - feat2);
}
//Find the most similar features in the database
string Facenet::faceRecognition(cv::Mat& feat, double threshold)
{
	int id = -1;
	double mins = 0.;
	for (int i = 0; i < _feat.size(); i++)
	{
		double simx = getSimilarity(_feat[i], feat);
		if (simx < mins || id == -1)
		{
			id = i;
			mins = simx;
		}
	}
	if (mins > threshold || id==-1)return "none";
	if (_nametolabel.count(_vimgname[id]) > 0)
		return _nametolabel[_vimgname[id]];

	return "none";
}