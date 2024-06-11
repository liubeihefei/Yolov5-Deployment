#include <openvino/openvino.hpp>
#include <string.h>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <algorithm>


typedef struct armor
{
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float score;
    int label;
} armor;


class detector
{
private:
    //推理模型
    ov::InferRequest infer_request;
    //缩放比例，xy方向padding
    float scale;
    int padding_y;
    int padding_x;

public:
    // 构造函数，构建模型
    detector(std::string path);

    // 细狗函数
    ~detector();

    // 图像遍历回调函数
    void pixelCallback(int y, int x, const cv::Vec3b& pixel, float* pixelData);

    // 预处理，得到输入tensor，image为输入图像，tensor为转换后的输入张量
    void preprocess(cv::Mat& image, ov::Tensor& tensor);

    // 计算iou
    float cal_iou(armor a, armor b);

    // 非极大值抑制
    void nms(float* result, float conf_thr, float iou_thr, std::vector<armor>& armors);

    // 确保roi在图内
    void preventExceed(int &x, int &y, int &width, int &height, const cv::Mat &src);

    // 从输入到结果封装的完整过程
    void detect(const cv::Mat& image, float conf_thr, float iou_thr, std::vector<armor>& armors);

    void detect_little(const cv::Mat& image, float conf_thr, float iou_thr, std::vector<armor>& armors, int nums);

    void detect_little_test(const cv::Mat& image, float conf_thr, float iou_thr, std::vector<armor>& armors);

    
    //-------------以下为测试功能------------

    // 角点纠正
    void correct_grid(std::vector<armor>& armors, const cv::Mat &src);

    // 先提取roi，再进行传统二值化+提取提取目标
    void find_ydd(cv::Mat& image, std::vector<armor>& armors);

    // 将图像进行4 * 3分割
    std::vector<cv::Mat> split_img(cv::Mat image, int nums);
};