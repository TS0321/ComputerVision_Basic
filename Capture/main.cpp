#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
int main(int argh, char* argv[])
{
    cv::VideoCapture cap(0);//デバイスのオープン
    //cap.open(0);//こっちでも良い．
    //cap.set(cv::CAP_PROP_FOCUS, 500);
    const int camera_width = 640;
    const int camera_height = 480;

    cap.set(cv::CAP_PROP_FRAME_WIDTH, camera_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height);

    if (!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
    {
        //読み込みに失敗したときの処理
        return -1;
    }

    cv::Mat frame; //取得したフレーム
    int img_num = 0;
    while (cap.read(frame))//無限ループ
    {
        //
        //取得したフレーム画像に対して，クレースケール変換や2値化などの処理を書き込む．
        //

        cv::imshow("win", frame);//画像を表示．
        const int key = cv::waitKey(1);
        if (key == 'q'/*113*/)//qボタンが押されたとき
        {
            break;//whileループから抜ける．
        }
        else if (key == 's'/*115*/)//sが押されたとき
        {
            //フレーム画像を保存する．
            std::string filename;
            filename = "../../../Capdata/img" + std::to_string(img_num++) + ".png";
            cv::imwrite(filename, frame);
            cv::waitKey(10);
            std::cout << filename << " is saved!" << std::endl;
        }
    }
    cv::destroyAllWindows();
    return 0;
}