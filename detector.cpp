#include "detector.h"


MTCNNDetector::MTCNNDetector(std::string modelpath)
{
    //Configure three networks
    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = modelpath + "/det1.caffemodel";
    pConfig.protoText = modelpath + "/det1.prototxt";
    pConfig.threshold = 0.6f;

    RefineNetwork::Config rConfig;
    rConfig.caffeModel = modelpath + "/det2.caffemodel";
    rConfig.protoText = modelpath + "/det2.prototxt";
    rConfig.threshold = 0.7f;

    OutputNetwork::Config oConfig;
    oConfig.caffeModel = modelpath + "/det3.caffemodel";
    oConfig.protoText = modelpath + "/det3.prototxt";
    oConfig.threshold = 0.7f;

    _pnet = std::unique_ptr<ProposalNetwork>(new ProposalNetwork(pConfig));
    _rnet = std::unique_ptr<RefineNetwork>(new RefineNetwork(rConfig));
    _onet = std::unique_ptr<OutputNetwork>(new OutputNetwork(oConfig));
}

std::vector<Face> MTCNNDetector::detect(const cv::Mat& img,
    const float minFaceSize, const float maxFacescale,
    const float scaleFactor) {
    if (img.empty()) {
        return std::vector<Face>();
    }
    cv::Mat rgbImg = convertImg(img);
    //Transposition is because the input of matcnn is ncwh, and the length and width are opposite
    rgbImg = rgbImg.t();
    //Run P candidate network to detect face preliminarily
    std::vector<Face> faces = _pnet->run(rgbImg, minFaceSize, maxFacescale,scaleFactor);
    // ³õ²½¼ì²âÎª¿ÕÍË³ö
    if (faces.empty()) {
        return faces;
    }
    //Call R refinement network to remove dummy face frame
    faces = _rnet->run(rgbImg, faces);

    
    if (faces.empty()) {
        return faces;
    }
    //Call the o network to further detect the face and output five key points
    faces = _onet->run(rgbImg, faces);
    //Exchange the coordinate axis, because the dimension of the opencv matrix is opposite to the coordinate dimension
    for (size_t i = 0; i < faces.size(); ++i) {
        std::swap(faces[i].bbox.x1, faces[i].bbox.y1);
        std::swap(faces[i].bbox.x2, faces[i].bbox.y2);
        for (int p = 0; p < NUM_PTS; ++p) {
            std::swap(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]);
        }
    }
    return faces;
}

//Get projection matrix
void getAffineMatrix(double* src_5pts, const double* dst_5pts, double* M)
{
    double src[10], dst[10];
    memcpy(src, src_5pts, sizeof(double) * 10);
    memcpy(dst, dst_5pts, sizeof(double) * 10);
    // initialization
    double ptmp[2];
    ptmp[0] = ptmp[1] = 0;
    for (int i = 0; i < 5; ++i) {
        ptmp[0] += src[i];
        ptmp[1] += src[5 + i];
    }
    ptmp[0] /= 5;
    ptmp[1] /= 5;
    for (int i = 0; i < 5; ++i) {
        src[i] -= ptmp[0];
        src[5 + i] -= ptmp[1];
        dst[i] -= ptmp[0];
        dst[5 + i] -= ptmp[1];
    }
    // determine init angle
    double dst_x = (dst[3] + dst[4] - dst[0] - dst[1]) / 2, dst_y = (dst[8] + dst[9] - dst[5] - dst[6]) / 2;
    double src_x = (src[3] + src[4] - src[0] - src[1]) / 2, src_y = (src[8] + src[9] - src[5] - src[6]) / 2;
    double theta = atan2(dst_x, dst_y) - atan2(src_x, src_y);
    // determine init scale
    double scale = sqrt(pow(dst_x, 2) + pow(dst_y, 2)) / sqrt(pow(src_x, 2) + pow(src_y, 2));
    double pts1[10];
    double pts0[2];
    double _a = sin(theta), _b = cos(theta);
    pts0[0] = pts0[1] = 0;
    for (int i = 0; i < 5; ++i) {
        pts1[i] = scale * (src[i] * _b + src[i + 5] * _a);
        pts1[i + 5] = scale * (-src[i] * _a + src[i + 5] * _b);
        pts0[0] += (dst[i] - pts1[i]);
        pts0[1] += (dst[i + 5] - pts1[i + 5]);
    }
    pts0[0] /= 5;
    pts0[1] /= 5;

    double sqloss = 0;
    for (int i = 0; i < 5; ++i) {
        sqloss += ((pts0[0] + pts1[i] - dst[i]) * (pts0[0] + pts1[i] - dst[i])
            + (pts0[1] + pts1[i + 5] - dst[i + 5]) * (pts0[1] + pts1[i + 5] - dst[i + 5]));
    }
    // optimization
    double square_sum = 0;
    for (int i = 0; i < 10; ++i) {
        square_sum += src[i] * src[i];
    }
    for (int t = 0; t < 200; ++t) {
        //cout << sqloss << endl;
        // determine angle
        _a = 0;
        _b = 0;
        for (int i = 0; i < 5; ++i) {
            _a += ((pts0[0] - dst[i]) * src[i + 5] - (pts0[1] - dst[i + 5]) * src[i]);
            _b += ((pts0[0] - dst[i]) * src[i] + (pts0[1] - dst[i + 5]) * src[i + 5]);
        }
        if (_b < 0) {
            _b = -_b;
            _a = -_a;
        }
        double _s = sqrt(_a * _a + _b * _b);
        _b /= _s;
        _a /= _s;

        for (int i = 0; i < 5; ++i) {
            pts1[i] = scale * (src[i] * _b + src[i + 5] * _a);
            pts1[i + 5] = scale * (-src[i] * _a + src[i + 5] * _b);
        }

        // determine scale
        double _scale = 0;
        for (int i = 0; i < 5; ++i) {
            _scale += ((dst[i] - pts0[0]) * pts1[i] + (dst[i + 5] - pts0[1]) * pts1[i + 5]);
        }
        _scale /= (square_sum * scale);
        for (int i = 0; i < 10; ++i) {
            pts1[i] *= (_scale / scale);
        }
        scale = _scale;

        // determine pts0
        pts0[0] = pts0[1] = 0;
        for (int i = 0; i < 5; ++i) {
            pts0[0] += (dst[i] - pts1[i]);
            pts0[1] += (dst[i + 5] - pts1[i + 5]);
        }
        pts0[0] /= 5;
        pts0[1] /= 5;

        double _sqloss = 0;
        for (int i = 0; i < 5; ++i) {
            _sqloss += ((pts0[0] + pts1[i] - dst[i]) * (pts0[0] + pts1[i] - dst[i])
                + (pts0[1] + pts1[i + 5] - dst[i + 5]) * (pts0[1] + pts1[i + 5] - dst[i + 5]));
        }
        if (abs(_sqloss - sqloss) < 1e-2) {
            break;
        }
        sqloss = _sqloss;
    }
    // generate affine matrix
    for (int i = 0; i < 5; ++i) {
        pts1[i] += (pts0[0] + ptmp[0]);
        pts1[i + 5] += (pts0[1] + ptmp[1]);
    }
    // printMat(pts1, 2, 5);
    M[0] = _b * scale;
    M[1] = _a * scale;
    M[3] = -_a * scale;
    M[4] = _b * scale;
    M[2] = pts0[0] + ptmp[0] - scale * (ptmp[0] * _b + ptmp[1] * _a);
    M[5] = pts0[1] + ptmp[1] - scale * (-ptmp[0] * _a + ptmp[1] * _b);
}
/// <summary>
/// Detect and align pictures in the dataset
/// </summary>
/// <param name="imgdir"></param>
/// <param name="savedir"></param>
void MTCNNDetector::datasetAlign(std::string imgdir, std::string savedir)
{
    std::vector<std::string> imgspath;
    cv::glob(imgdir + "/*.jpg", imgspath);
    for (int i = 0; i < imgspath.size(); i++)
    {
        int ed=0,st=0;
        for (int j = 0; j < imgspath[i].size(); j++)
        {
            if (imgspath[i][j] == '\\')imgspath[i][j] = '/';
            if (imgspath[i][j] == '.')ed = j;
            if (imgspath[i][j] == '/')st = j+1;
        }
        int len = ed - st;
        if (len <= 0)continue;
        cv::Mat img = cv::imread(imgspath[i]);
        std::vector<Face> faces = detect(img, 20.f, 1.f, 0.709f);
        faceAlign(img, faces, savedir + "/" + imgspath[i].substr(st, len));
    }
}


//Face alignment
void MTCNNDetector::faceAlign(cv::Mat& img, std::vector<Face>& faces, std::string savepath)
{
    double coord5point1[10] = { 30.2946, 65.5318, 48.0252, 33.5493, 62.7299, 
        51.6963, 51.5014, 71.7366, 92.3655, 92.2041 }; //112x96 target point
    double coord5point2[10] = { 30.2946 + 8.0000, 65.5318 + 8.0000, 
        48.0252 + 8.0000, 33.5493 + 8.0000, 62.7299 + 8.0000, 51.6963, 
        51.5014, 71.7366, 92.3655, 92.2041 }; //Target point of 112x112

    for (int i = 0; i < faces.size(); i++)
    {
        int x1 = faces[i].bbox.x1;
        int y1 = faces[i].bbox.y1;
        int x2 = faces[i].bbox.x2;
        int y2 = faces[i].bbox.y2;

        //Enlarge by 100% to prevent black edges on the face after alignment
        int new_x1 = std::max(int(1.50 * x1 - 0.50 * x2), 0);
        int new_x2 = std::min(int(1.50 * x2 - 0.50 * x1), img.cols - 1);
        int new_y1 = std::max(int(1.50 * y1 - 0.50 * y2), 0);
        int new_y2 = std::min(int(1.50 * y2 - 0.50 * y1), img.rows - 1);
        cv::Mat enlargedFace = img(cv::Rect(new_x1, new_y1, new_x2 - new_x1 + 1, new_y2 - new_y1 + 1));
        //imwrite("enlargedFace.jpg", enlargedFace);

        //Get the key points in the original figure
        int left_eye_x = faces[i].ptsCoords[0];
        int left_eye_y = faces[i].ptsCoords[1];
        int right_eye_x = faces[i].ptsCoords[2];
        int right_eye_y = faces[i].ptsCoords[3];
        int nose_x = faces[i].ptsCoords[4];
        int nose_y = faces[i].ptsCoords[5];
        int left_mouth_x = faces[i].ptsCoords[6];
        int left_mouth_y = faces[i].ptsCoords[7];
        int right_mouth_x = faces[i].ptsCoords[8];
        int right_mouth_y = faces[i].ptsCoords[9];

        //Get the coordinates of key points in the figure after 100% expansion (the purpose of expansion is to prevent black edges after alignment)
        int new_left_eye_x = left_eye_x - new_x1;
        int new_right_eye_x = right_eye_x - new_x1;
        int new_nose_x = nose_x - new_x1;
        int new_left_mouth_x = left_mouth_x - new_x1;
        int new_right_mouth_x = right_mouth_x - new_x1;
        int new_left_eye_y = left_eye_y - new_y1;
        int new_right_eye_y = right_eye_y - new_y1;
        int new_nose_y = nose_y - new_y1;
        int new_left_mouth_y = left_mouth_y - new_y1;
        int new_right_mouth_y = right_mouth_y - new_y1;

        double M[6];
        double enlargeFace_landmark[10] = { new_left_eye_x, new_right_eye_x, new_nose_x, new_left_mouth_x, new_right_mouth_x, new_left_eye_y, new_right_eye_y, new_nose_y, new_left_mouth_y, new_right_mouth_y };
        //112x112
        getAffineMatrix(enlargeFace_landmark, coord5point2, M);
        cv::Mat warp_mat_112x112 = (cv::Mat_<float>(2, 3) << M[0], M[1], M[2], M[3], M[4], M[5]);
        cv::Mat alignFace_112x112 = cv::Mat::zeros(112, 112, img.type());
        warpAffine(enlargedFace, alignFace_112x112, warp_mat_112x112, alignFace_112x112.size());
        cv::imwrite(savepath + "_align" + std::to_string(i) + ".jpg", alignFace_112x112);
    }

}