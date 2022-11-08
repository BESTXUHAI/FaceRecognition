#ifndef _include_opencv_mtcnn_detector_h_
#define _include_opencv_mtcnn_detector_h_

#include "face.h"
#include "onet.h"
#include "pnet.h"
#include "rnet.h"


class MTCNNDetector {
private:
    std::unique_ptr<ProposalNetwork> _pnet;
    std::unique_ptr<RefineNetwork> _rnet;
    std::unique_ptr<OutputNetwork> _onet;

public:
    MTCNNDetector(std::string modelpath);
    std::vector<Face> detect(const cv::Mat& img, const float minFaceSize, const float maxFacescale,
        const float scaleFactor);
    void faceAlign(cv::Mat& img, std::vector<Face>& faces, std::string savepath);
    void datasetAlign(std::string imgdir, std::string savedir);
};

#endif
