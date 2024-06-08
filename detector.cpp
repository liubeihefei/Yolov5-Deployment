#include "detector.hpp"
#include <time.h>
#include <cmath>  


detector::detector(std::string path)
{
    // 创建核心
    ov::Core core;
    // 读取模型
    std::shared_ptr<ov::Model> model = core.read_model(path);
    // 编译模型
    //ov::CompiledModel compiled_model = core.compile_model(model, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY), ov::hint::num_requests(16));
    // ov::CompiledModel compiled_model = core.compile_model(model, "MULTI:CPU,GPU.0");
    ov::CompiledModel compiled_model = core.compile_model(model, "GPU");
    // 生成推理模型eeee
    infer_request = compiled_model.create_infer_request();

}

detector::~detector(){};

void detector::pixelCallback(int y, int x, const cv::Vec3b& pixel, float* pixelData)
{
    // 将像素值存储到float*类型的变量中
    const float b = static_cast<float>(pixel[0]);
    const float g = static_cast<float>(pixel[1]);
    const float r = static_cast<float>(pixel[2]);


    // 假设像素数据是按顺序存储的（B, G, R, B, G, R, ...），需要归一化
    const int index = (y * 640 + x) ;
    pixelData[index] = b / 255;
    pixelData[index + 640 * 640] = g / 255;
    pixelData[index + 640 * 640 * 2] = r / 255;
}

void detector::preprocess(cv::Mat& image, ov::Tensor& tensor)
{
    auto data = tensor.data<float>();

    image.forEach<cv::Vec3b>(
            [&](cv::Vec3b & pixel, const int *position)
            {
                const int y = position[0];
                const int x = position[1];
                pixelCallback(y, x, pixel, data);
            });
}

float detector::cal_iou(armor a, armor b)
{
    int ax_min = std::min(std::min(std::min(a.x1, a.x2), a.x3), a.x4);
    int ax_max = std::max(std::max(std::max(a.x1, a.x2), a.x3), a.x4);
    int ay_min = std::min(std::min(std::min(a.y1, a.y2), a.y3), a.y4);
    int ay_max = std::max(std::max(std::max(a.y1, a.y2), a.y3), a.y4);

    int bx_min = std::min(std::min(std::min(b.x1, b.x2), b.x3), b.x4);
    int bx_max = std::max(std::max(std::max(b.x1, b.x2), b.x3), b.x4);
    int by_min = std::min(std::min(std::min(b.y1, b.y2), b.y3), b.y4);
    int by_max = std::max(std::max(std::max(b.y1, b.y2), b.y3), b.y4);

    float max_x = std::max(ax_min, bx_min);
    float min_x = std::min(ax_max, bx_max);
    float max_y = std::max(ay_min, by_min);
    float min_y = std::min(ay_max, by_max);

    if(min_x <= max_x || min_y <= max_y)
        return 0;
    
    float over_area = (min_x - max_x) * (min_y - max_y);

    float area_a = (ax_max - ax_min) * (ay_max - ay_min);
    float area_b = (bx_max - bx_min) * (by_max - by_min);
    float iou = over_area / (area_a + area_b - over_area);
    return iou;

}

void detector::nms(float* result, float conf_thr, float iou_thr, std::vector<armor>& armors)
{
    int res_nums = 25200;
    // int res_nums = 8400;
    //遍历result，如果conf大于阈值conf_thr，则放入armors
    for(int i = 0;i < res_nums;++i)
    {
        // float temp_s = result[8 + i * 4];
        // for(int k = i * 43 + 9;k < i * 43 + 43;++k)
        //     if(temp_s < result[k])
        //         temp_s = result[k];
        // if(temp_s >= conf_thr)
        if(result[8 + i * 10] >= conf_thr)
        {
            armor temp;
            //将四个角点放入
            temp.x1 = int((result[0 + i * 10] - padding_x) / scale);   temp.x2 = int((result[2 + i * 10] - padding_x) / scale);
            temp.x3 = int((result[4 + i * 10] - padding_x) / scale);   temp.x4 = int((result[6 + i * 10] - padding_x) / scale);
            temp.y1 = int((result[1 + i * 10] - padding_y) / scale);   temp.y2 = int((result[3 + i * 10] - padding_y) / scale);
            temp.y3 = int((result[5 + i * 10] - padding_y) / scale);   temp.y4 = int((result[7 + i * 10] - padding_y) / scale);

            //找到最大的条件类别概率并乘上conf作为类别概率
            float cls = result[i * 10 + 9];
            int cnt = 0;
            for(int j = i * 10 + 10;j < i * 10 + 10;++j)
            {
                if(cls < result[j])
                    {
                        cls = result[j];
                        cnt = (j - 9) % 10;
                    }
            }
            // cls *= result[8 + i * 10];
            // cls = temp_s;
            temp.score = cls;
            temp.label = cnt;
            armors.push_back(temp);
        }
    }
    
    //对得到的armor按score？进行降序排序（似乎应该按conf，但好像差不多）
    std::sort(armors.begin(), armors.end(), [](armor a, armor b) { return a.score > b.score; });

    //按iou_thr将重合度高的armor进行筛掉
    for(int i = 0;i < int(armors.size());++i)
    {
        for(int j = i + 1;j < int(armors.size());++j)
            //如果与当前的框iou大于阈值则erase掉
            if(cal_iou(armors[i], armors[j]) > iou_thr)
            {
                armors.erase(armors.begin() + j);
                --j;//万年不见--
            }
    }
}

void detector::preventExceed(int &x, int &y, int &width, int &height, const cv::Mat &src)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + width > src.cols) width = src.cols - x;
    if (y + height > src.rows) height = src.rows - y;
}

void detector::detect(const cv::Mat& image, float conf_thr, float iou_thr, std::vector<armor>& armors)
{
    //不要在原图上操作
    cv::Mat image0 = image;
    //计算缩放大小，padding个数，先缩放再padding
    scale = std::min(float(640) / image0.cols, float(640) / image0.rows);
    padding_y = int((640 - image0.rows * scale) / 2);
    padding_x = int((640 - image0.cols * scale) / 2);
    cv::resize(image0, image0, cv::Size(image0.cols * scale, image0.rows * scale), cv::INTER_LINEAR);
    cv::copyMakeBorder(image0, image0, padding_y, padding_y, padding_x, padding_x, cv::BORDER_CONSTANT, (144, 144, 144));
    //获得输入张量
    ov::Tensor input_tensor = infer_request.get_input_tensor(0);
    preprocess(image0, input_tensor);

    auto start = std::chrono::steady_clock::now();

    infer_request.infer();

    auto end = std::chrono::steady_clock::now();
    double dur = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "time: " << dur << std::endl;

    //推理得float*结果
    ov::Tensor output_tensor = infer_request.get_output_tensor(0);
    auto result = output_tensor.data<float>();

   
    //得到最后的装甲板
    nms(result, conf_thr, iou_thr, armors);
}

void detector::detect_little(const cv::Mat& image, float conf_thr, float iou_thr, std::vector<armor>& armors)
{
    //不要在原图上操作
    cv::Mat image0 = image;
    //对图像进行分割并串行推理
    std::vector<cv::Mat> imgs = split_img(image0);
    //一些必要信息
    int nums = std::sqrt(imgs.size());
    int height = image.rows;
    int wight = image.cols;
    int ws = wight / nums;
    int hs = height / nums;


    auto start = std::chrono::steady_clock::now();

    for(int i = 0;i < imgs.size();++i)
    {
        std::cout << i << std::endl;
        //计算缩放大小，padding个数，先缩放再padding
        scale = std::min(float(640) / imgs[i].cols, float(640) / imgs[i].rows);
        padding_y = int((640 - imgs[i].rows * scale) / 2);
        padding_x = int((640 - imgs[i].cols * scale) / 2);
        cv::resize(imgs[i], imgs[i], cv::Size(imgs[i].cols * scale, imgs[i].rows * scale), cv::INTER_LINEAR);
        cv::copyMakeBorder(imgs[i], imgs[i], padding_y, padding_y, padding_x, padding_x, cv::BORDER_CONSTANT, (144, 144, 144));
        //获得输入张量
        ov::Tensor input_tensor = infer_request.get_input_tensor(0);
        preprocess(imgs[i], input_tensor);
        infer_request.infer();
        //推理得float*结果
        ov::Tensor output_tensor = infer_request.get_output_tensor(0);
        auto result = output_tensor.data<float>();

        int before = armors.size();
        //得到最后的装甲板
        nms(result, conf_thr, iou_thr, armors);
        int after = armors.size();

        //对装甲板中的点进行处理
        for(int j = before;j < after;++j)
        {
            //先将点缩小到原本roi区域中
            armors[j].x1 /= nums;   armors[j].y1 /= nums;
            armors[j].x2 /= nums;   armors[j].y2 /= nums;
            armors[j].x3 /= nums;   armors[j].y3 /= nums;
            armors[j].x4 /= nums;   armors[j].y4 /= nums;
            //根据i，反推在原图中的哪个宫格
            int ori_xth = i / nums;
            int ori_yth = i % nums;
            armors[j].x1 += ori_xth * ws;   armors[j].y1 += ori_yth * hs;
            armors[j].x2 += ori_xth * ws;   armors[j].y2 += ori_yth * hs;
            armors[j].x3 += ori_xth * ws;   armors[j].y3 += ori_yth * hs;
            armors[j].x4 += ori_xth * ws;   armors[j].y4 += ori_yth * hs;
        }
    }

    auto end = std::chrono::steady_clock::now();
    double dur = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "time: " << dur << std::endl;
}

void detector::correct_grid(std::vector<armor>& armors, const cv::Mat &src)
{
    //遍历每个装甲板，对角点进行修正
    for(int n = 0;n < int(armors.size());++n)
    {
        cv::circle(src, cv::Point(armors[n].x1, armors[n].y1), 1, cv::Scalar(255,0,0),-1);
        cv::circle(src, cv::Point(armors[n].x2, armors[n].y2), 1, cv::Scalar(255,0,0),-1);
        cv::circle(src, cv::Point(armors[n].x3, armors[n].y3), 1, cv::Scalar(255,0,0),-1);
        cv::circle(src, cv::Point(armors[n].x4, armors[n].y4), 1, cv::Scalar(255,0,0),-1);


        int fu = 5;
        int pad = fu * 2;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));


        //对第一个角点进行修正，获取7 * 7区域
        int min_x = armors[n].x1 - fu;
        int min_y = armors[n].y1 - fu;
        int width = pad;
        int height = pad;
        preventExceed(min_x, min_y, width, height, src);
        cv::Mat temp_img1 = src(cv::Rect(min_x, min_y, width, height));
        // cv::imwrite("/home/horsefly/yolov5-deployment/save/img1_raw.jpg", temp_img1);
        //大津法进行明暗区分
        // cv::cvtColor(temp_img1, temp_img1, cv::COLOR_BGR2GRAY);
        cv::Mat channels[3];
        cv::split(temp_img1, channels);
        // cv::imwrite("/home/horsefly/yolov5-deployment/save/img1_grey.jpg", channels[2]);
        // cv::threshold(temp_img1, temp_img1, 0, 255, cv::THRESH_OTSU);
        cv::threshold(channels[2], temp_img1, 0, 255, cv::THRESH_OTSU);

        cv::erode(temp_img1, temp_img1, kernel);
        cv::dilate(temp_img1, temp_img1, kernel);
        // cv::imwrite("/home/horsefly/yolov5-deployment/save/img1_out.jpg", temp_img1);

        

        //遍历区域取左上第一个0->255的255边界点
        for (int i = 0; i < temp_img1.rows; i++)
        {
            int flag = false;
            for (int j = 0; j < temp_img1.cols; j++)
            {
                if(temp_img1.at<uchar>(i, j) == 255)
                {
                    // std::cout << i << " " << j << std::endl;
                    min_x += j;
                    min_y += i;
                    flag = true;
                    break;
                }
            }
            if(flag == true)
                break;
        }
        //修正角点
        if(min_x != armors[n].x1 - fu && min_y != armors[n].y1 - fu)
        {
            armors[n].x1 = min_x;
            armors[n].y1 = min_y;
        }


        //对第二个角点进行修正，获取7 * 7区域
        min_x = armors[n].x2 - fu;
        min_y = armors[n].y2 - fu;
        width = pad;
        height = pad;
        preventExceed(min_x, min_y, width, height, src);
        cv::Mat temp_img2 = src(cv::Rect(min_x, min_y, width, height));
        cv::imwrite("/home/horsefly/yolov5-deployment/save/img1_raw.jpg", temp_img2);
        //大津法进行明暗区分
        cv::split(temp_img2, channels);
        cv::imwrite("/home/horsefly/yolov5-deployment/save/img1_grey.jpg", channels[2]);
        // cv::cvtColor(temp_img2, temp_img2, cv::COLOR_BGR2GRAY);
        cv::threshold(channels[2], temp_img2, 0, 255, cv::THRESH_OTSU);
        cv::erode(temp_img2, temp_img2, kernel);
        cv::dilate(temp_img2, temp_img2, kernel);
        cv::imwrite("/home/horsefly/yolov5-deployment/save/img1_out.jpg", temp_img2);

        //遍历区域取左下第一个0->255的255边界点
        for (int i = temp_img2.rows - 1; i >= 0; i--)
        {
            int flag = false;
            for (int j = 0; j <= temp_img2.cols - 1; j++)
            {
                if(temp_img2.at<uchar>(i, j) == 255)
                {
                    min_x += j;
                    min_y += i;
                    flag = true;
                    break;
                }
            }
            if(flag == true)
                break;
        }
        //修正角点
        if(min_x != armors[n].x2 - fu && min_y != armors[n].y2- fu)
        {
            armors[n].x2 = min_x;
            armors[n].y2 = min_y;
        }


        //对第三个角点进行修正，获取7 * 7区域
        min_x = armors[n].x3 - fu;
        min_y = armors[n].y3 - fu;
        width = pad;
        height = pad;
        preventExceed(min_x, min_y, width, height, src);
        cv::Mat temp_img3 = src(cv::Rect(min_x, min_y, width, height));
        //大津法进行明暗区分
        cv::split(temp_img3, channels);
        // cv::cvtColor(temp_img3, temp_img3, cv::COLOR_BGR2GRAY);
        cv::threshold(channels[2], temp_img3, 0, 255, cv::THRESH_OTSU);
        cv::erode(temp_img3, temp_img3, kernel);
        cv::dilate(temp_img3, temp_img3, kernel);

        //遍历区域取右下第一个0->255的255边界点
        for (int i = temp_img3.rows - 1; i >= 0; i--)
        {
            int flag = false;
            for (int j = temp_img3.cols - 1; j >= 0; j--)
            {    
                if(temp_img3.at<uchar>(i, j) == 255)
                {
                    min_x += j;
                    min_y += i;
                    flag = true;
                    break;
                }
            }
            if(flag == true)
                break;
        }
        //修正角点
        if(min_x != armors[n].x3 - fu && min_y != armors[n].y3 - fu)
        {
            armors[n].x3 = min_x;
            armors[n].y3 = min_y;
        }


        //对第四个角点进行修正，获取7 * 7区域
        min_x = armors[n].x4 - fu;
        min_y = armors[n].y4 - fu;
        width = pad;
        height = pad;
        preventExceed(min_x, min_y, width, height, src);
        cv::Mat temp_img4 = src(cv::Rect(min_x, min_y, width, height));
        //大津法进行明暗区分
        cv::split(temp_img4, channels);
        // cv::cvtColor(temp_img4, temp_img4, cv::COLOR_BGR2GRAY);
        cv::threshold(channels[2], temp_img4, 0, 255, cv::THRESH_OTSU);
        cv::erode(temp_img4, temp_img4, kernel);
        cv::dilate(temp_img4, temp_img4, kernel);

        //遍历区域取右上第一个0->255的255边界点
        for (int i = 0; i <= temp_img4.rows - 1; i++)
        {
            int flag = false;
            for (int j = temp_img4.cols - 1;j >= 0; j--)
            { 
                if(temp_img4.at<uchar>(i, j) == 255)
                {
                    min_x += j;
                    min_y += i;
                    flag = true;
                    break;
                }
            }
            if(flag == true)
                break;
        }
        //修正角点
        if(min_x != armors[n].x4 - fu && min_y != armors[n].y4 - fu)
        {
            armors[n].x4 = min_x;
            armors[n].y4 = min_y;
        }

        cv::circle(src, cv::Point(armors[n].x1, armors[n].y1), 1, cv::Scalar(0,0,255),-1);
        cv::circle(src, cv::Point(armors[n].x2, armors[n].y2), 1, cv::Scalar(0,0,255),-1);
        cv::circle(src, cv::Point(armors[n].x3, armors[n].y3), 1, cv::Scalar(0,0,255),-1);
        cv::circle(src, cv::Point(armors[n].x4, armors[n].y4), 1, cv::Scalar(0,0,255),-1);
    }
}

void detector::find_ydd(cv::Mat& image, std::vector<armor>& armors)
{
    // 将非引导灯筛掉
    std::vector<armor> new_armors;
    for(int i = 0;i < armors.size();++i)
        if(armors[i].label == 33)
            new_armors.push_back(armors[i]);
    
    // 没有则返回
    if(new_armors.size() == 0)
        return;

    // 获得图像中心
    float img_cx = image.cols / 2;
    float img_cy = image.rows / 2;

    // 只挑出距离图像中心最近的ydd
    float min_dis = 99999.0;
    int key = 0;
    for(int i = 0;i < new_armors.size();++i)
    {   
        // 计算中心与距离
        float temp_x = (new_armors[i].x1 + new_armors[i].x2 + new_armors[i].x3 + new_armors[i].x4) / 4;
        float temp_y = (new_armors[i].y1 + new_armors[i].y2 + new_armors[i].y3 + new_armors[i].y4) / 4;
        float temp_dis = std::pow(std::pow((img_cx - temp_x), 2) + std::pow((img_cy - temp_y), 2), 0.5);

        // 如果发现更近的则更新
        if(temp_dis < min_dis)
        {
            min_dis = temp_dis;
            key = i;
        }
    }

    // 选择最近的ydd，取bbox
    armor final_ydd = new_armors[key];
    int min_x = std::min(final_ydd.x4, std::min(final_ydd.x3, std::min(final_ydd.x1, final_ydd.x2)));
    int min_y = std::min(final_ydd.y4, std::min(final_ydd.y3, std::min(final_ydd.y1, final_ydd.y2)));
    int max_x = std::max(final_ydd.x4, std::max(final_ydd.x3, std::max(final_ydd.x1, final_ydd.x2)));
    int max_y = std::max(final_ydd.y4, std::max(final_ydd.y3, std::max(final_ydd.y1, final_ydd.y2)));
    int ydd_width = max_x - min_x;
    int ydd_height = max_y - min_y;
    preventExceed(min_x, min_y, ydd_width, ydd_height, image);
    cv::Mat roi = image(cv::Rect(min_x, min_y, ydd_width, ydd_height));

    // roi为空也返回
    if(roi.cols == 0 || roi.rows == 0)
        return;

    // 对roi区域进行自适应二值化，进行开操作
    cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
    cv::threshold(roi, roi, 0, 255, cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::dilate(roi, roi, kernel);
    cv::erode(roi, roi, kernel);

    // 找轮廓，得到外接矩形
    std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i>hierarchy;
    cv::findContours(roi, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(min_x, min_y));
    std::vector<cv::Rect> boundrect(contours.size());
    for(int i = 0;i < contours.size();++i)
        // 获取轮廓外接矩形
        boundrect[i] = cv::boundingRect(contours[i]);

    // 选出面积最大的
    float max_area = 0.0;
    int key_area = 0;
    for(int i = 0;i < boundrect.size();++i)
    {
        if(max_area < boundrect[i].height * boundrect[i].width)
        {
            max_area = boundrect[i].height * boundrect[i].width;
            key_area = i;
        }
    }

    // 将找出的ydd画出来
    cv::rectangle(image, boundrect[key_area], cv::Scalar(0, 255, 0));
    cv::circle(image, cv::Point(boundrect[key_area].x + boundrect[key_area].width / 2, boundrect[key_area].y + boundrect[key_area].height / 2), 2, cv::Scalar(0, 0, 255), -1);
    cv::imshow("ydd_roi", roi);
}

std::vector<cv::Mat> detector::split_img(cv::Mat image)
{
    // debug开关
    bool debug = false;

    // 单方向上区分个数
    int nums = 8;

    // 获取图像宽高
    int width = image.cols;
    int height = image.rows;
    int ws = width / nums;
    int hs = height / nums;

    // 在原图上绘制宫格并保存
    if(debug)
    {
        cv::Mat boxes = image.clone();
        // 画线
        for(int i = 1;i < nums;++i)
        {
            cv::line(boxes, cv::Point(0, i * hs), cv::Point(width, i * hs), cv::Scalar(0, 255, 0), 1);
            cv::line(boxes, cv::Point(i * ws, 0), cv::Point(i * ws, height), cv::Scalar(0, 255, 0), 1);
        }
        cv::imwrite("/home/horsefly/Yolov5-Deployment/save/origin.jpg", boxes);        
    }

    // 对图像进行分割
    std::vector<cv::Mat> imgs;
    for(int i = 0;i < nums;++i)
    {
        for(int j = 0;j < nums;++j)
        {
            // 打印
            if(debug)
                std::cout << ws * i << " " <<  hs * j << " " << ws << " " << hs << std::endl;
            // 定义矩形roi
            cv::Rect tkw(ws * i, hs * j, ws, hs);
            // 进行roi提取，放大
            cv::Mat temp;
            cv::resize(image(tkw), temp, cv::Size(width, height), 0, 0, cv::INTER_AREA);
            // 放入imgs
            imgs.push_back(temp);
        }
    }

    // 保存图片
    if(debug)
        for(int i = 0;i < imgs.size();++i)
            cv::imwrite("/home/horsefly/Yolov5-Deployment/save/" + std::to_string(i) + ".jpg", imgs[i]);

    return imgs;
}