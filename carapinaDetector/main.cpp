#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <map>
#include <fstream>

#include "extra.hpp"
#include "cnn.hpp"
#include "cnn_config.hpp"

using namespace std;

const std::string DATASET_PATH = "D:\\data";

void doClucterization(map<int, map<int, pair<int, int>>>& points, int windowSizeRadius = 16) {

    for (auto& xIt : points) {
        for (auto& yIt : xIt.second) {
            pair<int, int> center;

            // while центр не смещается

            vector<pair<int, int>> rectZone;

            // получаем точки, лежащие в определённом диапазоне
            int leftBorder = xIt.first - windowSizeRadius;  //xIt.first - значение при инициализации (до while)
            int rightBorder = xIt.first + windowSizeRadius;
            int upperBorder = yIt.first - windowSizeRadius;  //yIt.first - значение при инициализации (до while)
            int bottomBorder = yIt.first + windowSizeRadius;

            auto _xrit = map<int, map<int, pair<int, int>>>::reverse_iterator(points.find(xIt.first));
            for (; _xrit != points.rend() && _xrit->first >= leftBorder; ++_xrit) { // === влево
                auto& interv = _xrit->second;
                auto _yrit = map<int, pair<int, int>>::reverse_iterator(interv.find(xIt.first));
                for (; _yrit != interv.rend() && _yrit->first >= upperBorder; ++_yrit) { // вверх
                    rectZone.push_back(_yrit->second);
                }
                auto _yit = interv.find(xIt.first);
                if (_yit != interv.end())
                    ++_yit;
                for (; _yit != interv.end() && _yit->first <= bottomBorder; ++_yrit) { // вниз
                    rectZone.push_back(_yit->second);
                }
            }
            auto _xit = points.find(xIt.first);
            if (_xit != points.end())
                ++_xit;
            for (; _xit != points.end() && _xit->first <= rightBorder; ++_xit) { // === вправо
                auto& interv = _xit->second;
                auto _yrit = map<int, pair<int, int>>::reverse_iterator(interv.find(xIt.first));
                for (; _yrit != interv.rend() && _yrit->first >= upperBorder; ++_yrit) { // вверх
                    rectZone.push_back(_yrit->second);
                }
                auto _yit = interv.find(xIt.first);
                if (_yit != interv.end())
                    ++_yit;
                for (; _yit != interv.end() && _yit->first <= bottomBorder; ++_yrit) { // вниз
                    rectZone.push_back(_yit->second);
                }
            }

            double koeff = 1.0 / rectZone.size();
            double cx = 0;
            double cy = 0;
            for (auto& it : rectZone) {
                cx += (double)it.first * koeff;
                cy += (double)it.second * koeff;
            }
            center.first = (int)cx;
            center.second = (int)cy;

            cv::Mat a;
            a.release();
        }
    }

}

void processGrayMat(cv::Mat& mat, int c, const std::string& path/*, CNN_Controller& cnnc*/) {
    using namespace cv;
    using namespace std;
    sf::Clock timer;
    timer.restart();

    Mat blured;
    int bSize = 7;
    blur(mat, blured, Size(bSize, bSize));
    Canny(blured, blured, 20, 50, 3);

    string catalog = "D:\\data\\out\\" + to_string(c);
    filesystem::create_directories(catalog);

    Mat rgb;
    cvtColor(mat, rgb, COLOR_GRAY2RGB);
    // Probabilistic Line Transform
    vector<cv::Vec4i> linesP; // will hold the results of the detection
    cv::HoughLinesP(blured, linesP, 2, CV_PI / 90, 30, 40, 15); // runs the actual detection
    // Draw the lines
    for (size_t i = 0; i < linesP.size(); i++)
    {
        cv::Vec4i l = linesP[i];
        cv::line(rgb, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        int cx = (l[0] + l[2]) / 2;
        int cy = (l[1] + l[3]) / 2;
        int x0 = cx - 16;
        int y0 = cy - 16;
        int x1 = cx + 17;
        int y1 = cy + 17;
        if (x0 < 0 || y0 < 0 || x1 >= mat.size().width || y1 >= mat.size().height)
            continue;
        Rect2i crop = Rect2i(Point2i(x0, y0), Point2i(x1, y1));
        imwrite(catalog + "\\" + to_string(i) + ".bmp", mat(crop));
    }
    imwrite("D:\\data\\out\\" + to_string(c) + ".jpg", rgb);

    cvtColor(rgb, rgb, COLOR_RGB2BGR);

    mat.release();
    blured.release();
    mat = rgb;
    

    //// === размываем изображение
    //Mat blured;
    //int gSize = 15;
    ////GaussianBlur(mat, blured, { gSize, gSize }, double(gSize) / 2);
    //blur(mat, blured, Size(7, 7));
    //Mat gradX, gradY, absGradX, absGradY, grad;

    ///*
    //// === проходим фильтром собеля по X и по Y; полученные матрицы совмещаем
    //int sobelKSize = 3;
    //double scale = 1.0;
    //double delta = 0.0;
    //int borderType = BORDER_DEFAULT;
    //// выходная матрица имеет размерность 16
    //Sobel(blured, gradX, CV_16S, 1, 0, sobelKSize, scale, delta, borderType);
    //Sobel(blured, gradY, CV_16S, 0, 1, sobelKSize, scale, delta, borderType);
    //// приводим их к размерности 8
    //convertScaleAbs(gradX, absGradX);
    //convertScaleAbs(gradY, absGradY);

    //addWeighted(absGradX, 0.5, absGradX, 0.5, 0, grad);

    //*/

    //Canny(blured, blured, 30, 70, 3);

    //// левый верхний угол интересных областей
    //map<int, map<int, bool>> contours = {};
    //for (int i = 31; i < grad.rows - 32; i++)
    //    for (int j = 31; j < grad.cols - 32; j++)
    //        if (grad.at<uchar>(i, j) == 255) {
    //            contours[j - 31][i - 31] = true;
    //        }

    //
    //map<int, map<int, pair<int, int>>> clusterCenter = {}; // на что среагировала сетка

    //// === ищем царапины сеткой
    //vector<Rect2i> crops;
    //for (auto& xIt : contours) {
    //    for (auto& yIt : xIt.second) {
    //        crops.emplace_back(xIt.first, yIt.first, 64, 64);
    //    }
    //}
    ///*
    //int save = 0;
    //int imgcntr = 0;
    //int stored = 0;
    //int border = cnnc.getMaxThreads() * 64;
    //vector<cnn::Tensor> input;
    //// обработатываем crops.size()/border точек
    //for (size_t i = 0; i < crops.size(); i++) {
    //    input.emplace_back(CNN_Controller::matToTensor(mat(crops[i])));
    //    stored++;
    //    if (stored >= border) {
    //        auto result = cnnc.forward(input);
    //        input.clear();

    //        for (size_t j = 0, ii = i + 1 - stored; j < result.size(); j++, ii++) {
    //            Rect2i& crop = crops[ii];
    //            if (save) {
    //                imwrite("D:\\net\\" + std::to_string(imgcntr) + ".png", mat(crop));
    //                imgcntr++;
    //            }
    //            save = 1 - save;
    //            if (result[j][0] <= 0.8)
    //                grad.at<uchar>(crop.y + 31, crop.x + 31) = 0;
    //            else
    //                clusterCenter[crop.x + 31][crop.y + 31] = { crop.x + 31, crop.y + 31 };
    //        }
    //        stored = 0;
    //    }
    //}
    //// обработатываем оставшиеся crops.size()%border точек
    //if (stored > 0) {
    //    auto result = cnnc.forward(input);
    //    input.clear();
    //    for (size_t j = 0, ii = crops.size() - stored; j < result.size(); j++, ii++) {
    //        Rect2i& crop = crops[ii];
    //        if (save) {
    //            imwrite("D:\\net\\" + std::to_string(imgcntr) + ".png", mat(crop));
    //            imgcntr++;
    //        }
    //        save = 1 - save;
    //        if (result[j][0] <= 0.8)
    //            grad.at<uchar>(crop.y + 31, crop.x + 31) = 0;
    //        else
    //            clusterCenter[crop.x + 31][crop.y + 31] = { crop.x + 31, crop.y + 31 };
    //    }
    //}
    //*/
    //
    //cnn::Tensor t;
    //Rect2i crop(0, 0, 64, 64);
    //for (auto& xIt : contours) {
    //    crop.x = xIt.first;
    //    for (auto& yIt : xIt.second) {
    //        crop.y = yIt.first;
    //        t = forward(matToTensor(mat(crop)));
    //        if (t[0] <= 0.8)
    //            grad.at<uchar>(crop.y + 31, crop.x + 31) = 0;
    //        else
    //            clusterCenter[crop.x + 31][crop.y + 31] = { crop.x + 31 , crop.y + 31 };
    //    }
    //}
    //

    //// Probabilistic Line Transform
    //grayMat.release();
    //grayMat = cv::Mat(filtered.rows, filtered.cols, CV_8UC1);
    //cv::Mat mat;
    //cv::cvtColor(grayMat, mat, cv::COLOR_GRAY2RGB);
    //vector<cv::Vec4i> linesP; // will hold the results of the detection
    //cv::HoughLinesP(filtered, linesP, 2, CV_PI / 90, 30, 50, 10); // runs the actual detection
    //// Draw the lines
    //for (size_t i = 0; i < linesP.size(); i++)
    //{
    //    cv::Vec4i l = linesP[i];
    //    cv::line(mat, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    //}

    //// === кластеризация
    ////doClucterization(clusterCenter, 16);


    //gradX.release();
    //gradY.release();
    //absGradX.release();
    //absGradY.release();
    //blured.release();
    //mat.release();

    //mat = grad;

    cout << "Processed in " << timer.getElapsedTime().asMilliseconds() << "ms\n";
}

int main()
{
    using namespace std;

    const int WIN_WIDTH = 1200;
    const int WIN_HEIGHT = 700;

    vector<string> imagePath = {};

    extra::loadFilenames(DATASET_PATH, ".bmp", imagePath);
    //for (auto& it : imagePath)
    //    cout << it << '\n';

    if (imagePath.empty()) {
        cout << "Directory \"" << DATASET_PATH << "\" is empty or doesn't exist.\n";
        return 0;
    }

    // загрузка весов сетки
    fstream f("weights23.json", ios::in);
    nlohmann::json weights;
    f >> weights;
    /*
    CNN_Controller controller;
    controller.initFromJson(weights);
    */
    loadFromJson(weights);
    f.close();

    sf::ContextSettings settings;
    settings.antialiasingLevel = 16;
    sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Carapina detector", sf::Style::Close, settings);
    sf::Clock timer;
    sf::Image image;
    sf::Texture texture;
    sf::Sprite sprite;
    sf::Font font;
    font.loadFromFile("C:\\Windows\\Fonts\\courbd.ttf");
    sf::Text text;
    text.setString(imagePath[0]);
    text.setFont(font);
    text.setFillColor(sf::Color::White);
    text.setOutlineThickness(1.5);
    text.setOutlineColor(sf::Color::Black);
    text.setCharacterSize(28);

    cv::Mat grayMat = cv::imread(imagePath[0], cv::IMREAD_GRAYSCALE);
    cout << "Showed file \"" << imagePath[0] << "\"\n";
    processGrayMat(grayMat, 0, imagePath[0]/*, controller*/);
    //cv::imwrite(imagePath[0] + ".jpg", grayMat);

    extra::cvtRGBMatToImage(grayMat, image);
    texture.loadFromImage(image);
    sprite.setTexture(texture);

    float scale = float(WIN_HEIGHT) / sprite.getLocalBounds().height;
    sprite.setScale({ scale, scale });

    int counter = 1;
    const int imgDelayMillis = 0;

    timer.restart();
    while (window.isOpen())
    {
        sf::Event e;
        while (window.pollEvent(e))
        {
            switch (e.type) {
            case sf::Event::Closed:
                window.close();
                break;
            case sf::Event::KeyPressed:
                if (e.key.code == sf::Keyboard::Escape)
                    window.close();
                break;
            }
        }

        if (timer.getElapsedTime().asMilliseconds() >= imgDelayMillis) {
            if (counter < imagePath.size()) {
                grayMat.release();
                grayMat = cv::imread(imagePath[counter], cv::IMREAD_GRAYSCALE);
                cout << "Showed file \"" << imagePath[counter] << "\"\n";
                processGrayMat(grayMat, counter, imagePath[counter]/*, controller */ );
                //cv::imwrite(imagePath[counter] + ".jpg", grayMat);

                extra::cvtRGBMatToImage(grayMat, image);
                texture.loadFromImage(image);

                text.setString(imagePath[counter]);

                counter++;
            }
            else text.setString("Finished");
            timer.restart();
        }

        window.clear();

        sprite.setTexture(texture);
        window.draw(sprite);

        window.draw(text);
        window.display();

        this_thread::sleep_for(chrono::milliseconds(1));
    }

    return 0;
}