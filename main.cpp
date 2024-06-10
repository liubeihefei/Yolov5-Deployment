#include "detector.hpp"
#include "video_player.hpp"
#include <time.h>

//颜色列表
const int color_list[80][3] ={
                {255 , 255 , 255},
                {236 ,176 , 31},
                {125 , 46 ,141},
                {118 ,171 , 47},
                { 76 ,189 ,237},
                {238 , 19 , 46},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };

//画armor
void draw_armor(cv::Mat& image, std::vector<armor>& armors, int nums)
{
    static const char* class_names[] = {
                                        "gl",
                                        // "armor_sentry_red", 
                                        "armor_sentry_blue",
                                        "armor_sentry_none",

                                        "armor_hero_red",
                                        "armor_hero_blue",
                                        "armor_hero_none",

                                        "armor_engine_red",
                                        "armor_engine_blue",
                                        "armor_engine_none",

                                        "armor_infantry_3_red",
                                        "armor_infantry_3_blue",
                                        "armor_infantry_3_none",

                                        "armor_infantry_4_red",
                                        "armor_infantry_4_blue",
                                        "armor_infantry_4_none",

                                        "armor_infantry_5_red",
                                        "armor_infantry_5_blue",
                                        "armor_infantry_5_none",

                                        "armor_outpost_red",
                                        "armor_outpost_blue",
                                        "armor_outpost_none",

                                        "armor_base_red",
                                        "armor_base_blue",
                                        "B3r",
                                        "B3b",
                                        "B3n",
                                        "B4r",
                                        "B4b",
                                        "B4n",
                                        "B5r",
                                        "B5b",
                                        "B5n",

                                        "bp",
                                        "ydd"
    };

    for (int i = 0; i < armors.size(); ++i)
    {
        //获取颜色
        cv::Scalar color = cv::Scalar(color_list[armors[i].label][0], color_list[armors[i].label][1], color_list[armors[i].label][2]);

        //绘制装甲板
        cv::line(image, cv::Point(armors[i].x1, armors[i].y1), cv::Point(armors[i].x2, armors[i].y2), color);
        cv::line(image, cv::Point(armors[i].x2, armors[i].y2), cv::Point(armors[i].x3, armors[i].y3), color);
        cv::line(image, cv::Point(armors[i].x3, armors[i].y3), cv::Point(armors[i].x4, armors[i].y4), color);
        cv::line(image, cv::Point(armors[i].x4, armors[i].y4), cv::Point(armors[i].x1, armors[i].y1), color);

        //标记类别和概率
        cv::putText(image, class_names[armors[i].label], cv::Point(armors[i].x1 + 28, armors[i].y1 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color);
        int score_temp = int(armors[i].score * 100);
        cv::putText(image, "0." + std::to_string(score_temp), cv::Point(armors[i].x1, armors[i].y1 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color);
    }

    // 获取图像宽高
    int width = image.cols;
    int height = image.rows;
    int ws = width / nums;
    int hs = height / nums;

    // 在原图上绘制宫格并保存
    for(int i = 1;i < nums;++i)
    {
        cv::line(image, cv::Point(0, i * hs), cv::Point(width, i * hs), cv::Scalar(0, 255, 0), 1);
        cv::line(image, cv::Point(i * ws, 0), cv::Point(i * ws, height), cv::Scalar(0, 255, 0), 1);
    }       
}

//用图像测试
int img_demo(std::string path, std::string model_path, std::string font, int num)
{
    //导入图片
    cv::Mat image;
    image = cv::imread(path);

    std::vector<armor> armors;
    detector new_detcetor(model_path);

    //检测与绘制
    new_detcetor.detect_little(image, 0.5, 0.4, armors, num);
    // new_detcetor.find_ydd(image, armors);
    draw_armor(image, armors, num);

    cv::imwrite("/home/horsefly/Yolov5-Deployment/save/out/" + font + "_" + std::to_string(num) + ".jpg", image);

    // new_detcetor.split_img(image);

    return 0;
}

//用视频测试
int video_demo(std::string path, std::string model_path, video_player video_player)
{
    cv::Mat image;
    cv::VideoCapture cap(path);
    detector new_detcetor(model_path);

    cv::Size sizeReturn = cv::Size(cap.get(3),cap.get(4));
    cv::VideoWriter writer("output.avi",cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),60,sizeReturn);

    // for(int i = 0;i < cap.get(7);++i)
    while(1)
    {
        cap >> image;

        //开始计时
        auto start = std::chrono::steady_clock::now();

        std::vector<armor> armors;
        // new_detcetor.detect(image, 0.1, 0.4, armors);

        // if(int(armors.size() > 0))
        //     new_detcetor.correct_grid(armors, image);

        // new_detcetor.find_ydd(image, armors);
        // draw_armor(image, armors);

        //结束计时
        auto end = std::chrono::steady_clock::now();
        double dur = std::chrono::duration<double, std::milli>(end - start).count();
        // cv::putText(image, std::to_string(int(1 / dur * 1000)), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0));

        writer << image;

        
        cv::imshow("Img", image);
        video_player.play_video(image);
    }
    return 0;
}

int main()
{
    
    std::string model_path = "/home/horsefly/Yolov5-Deployment/models/gl2.0/last.xml";

    // std::string video_path = "/media/horsefly/Ubuntu 22.0/zhefang/hero/2024-5-16-11_21_42/0.avi";
    // video_player player;
    // video_demo(video_path, model_path, player);


    // 进行11张图片1-10倍分割测试
    for(int i = 0;i < 11;++i)
    {
        std::string font = std::to_string(i);
        std::string img_path = "/home/horsefly/Yolov5-Deployment/save/gl_test/gl_test_" + font + ".jpg";
        int nums[] = {1, 2, 4, 8, 10};
        for(int j = 0;j < 5;++j)
            img_demo(img_path, model_path, font, nums[j]);
    }
    
    return 0;
}
