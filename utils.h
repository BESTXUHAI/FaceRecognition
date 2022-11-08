#ifndef _include_utils_h_
#define _include_utils_h_

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;
//Crop Image
inline cv::Mat cropImage(const cv::Mat& img, cv::Rect r) {
    cv::Mat m = cv::Mat::zeros(r.height, r.width, img.type());
    int dx = std::abs(std::min(0, r.x));
    if (dx > 0) {
        r.x = 0;
    }
    r.width -= dx;
    int dy = std::abs(std::min(0, r.y));
    if (dy > 0) {
        r.y = 0;
    }
    r.height -= dy;
    int dw = std::abs(std::min(0, img.cols - 1 - (r.x + r.width)));
    r.width -= dw;
    int dh = std::abs(std::min(0, img.rows - 1 - (r.y + r.height)));
    r.height -= dh;
    if (r.width > 0 && r.height > 0) {
        img(r).copyTo(m(cv::Range(dy, dy + r.height), cv::Range(dx, dx + r.width)));
    }
    return m;
}

//Convert image to model input
inline cv::Mat convertImg(const cv::Mat& img) {
    cv::Mat rgbImg;
    if (img.channels() == 3) {
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
    }
    else if (img.channels() == 4) {
        cv::cvtColor(img, rgbImg, cv::COLOR_BGRA2RGB);
    }

    rgbImg.convertTo(rgbImg, CV_32FC3);
    return rgbImg;
}
//tf.image.per_image_standardization
inline cv::Mat imgStandardization(const cv::Mat& img)
{
    cv::Mat imgs;
    double sizen = (double)img.rows * img.cols * 3;
    cv::Scalar means = cv::mean(img);
    double mean = (means[0] + means[1] + means[2]) / 3;
    double std = cv::norm(img - mean) / sqrt(sizen);
    double adjusted_std = cv::max(std, 1.0 / sqrt(sizen));
    imgs = (img - mean) / adjusted_std;
    return imgs;
}



#endif