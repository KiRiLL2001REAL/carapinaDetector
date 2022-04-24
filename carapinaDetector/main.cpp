#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

#include "extra.hpp"
#include "cnn.hpp"
#include "cnn_config.hpp"

const std::string DATASET_PATH = "D:\\Dataset";

void processGrayMat(cv::Mat& mat) {
    using namespace cv;
    using namespace std;
    sf::Clock timer;
    timer.restart();

    // === размываем изображение
    Mat blured;
    int gSize = 15;
    GaussianBlur(mat, blured, { gSize, gSize }, double(gSize) / 2);
    Mat gradX, gradY, absGradX, absGradY, grad;

    // === проходим фильтром собеля по X и по Y; полученные матрицы совмещаем
    int sobelKSize = 3;
    double scale = 1.0;
    double delta = 0.0;
    int borderType = BORDER_DEFAULT;
    // выходная матрица имеет размерность 16
    Sobel(blured, gradX, CV_16S, 1, 0, sobelKSize, scale, delta, borderType);
    Sobel(blured, gradY, CV_16S, 0, 1, sobelKSize, scale, delta, borderType);
    // приводим их к размерности 8
    convertScaleAbs(gradX, absGradX);
    convertScaleAbs(gradY, absGradY);

    Mat filteredX, filteredY;
    extra::filterStrong(absGradX, filteredX, 'x');
    extra::filterStrong(absGradY, filteredY, 'y');

    addWeighted(filteredX, 0.5, filteredY, 0.5, 0, grad);

    // === вычисляем пороговое значение для theshold
    long long counts[256]; memset(counts, 0, sizeof(long long) * 256);
    for (auto pointer = grad.datastart; pointer != grad.dataend; pointer++) {
        counts[*pointer]++;
    }
    for (int i = 0; i < 4; i++)
        counts[i] = 0;
    long long sum = 0;
    for (int i = 0; i < 256; i++)
        sum += counts[i];

    long long sumLocal = 0;
    int minPixel = 256;
    double koeff = 0.10;
    while (sumLocal < sum * koeff)
        sumLocal += counts[--minPixel];

    // === используя ранее найденное пороговое значение, преобразуем матрицу
    threshold(grad, grad, minPixel, 255, THRESH_BINARY);

    // === ищем царапины сеткой
    Rect2i crop(0, 0, 64, 64);
    for (int i = 31; i < grad.rows - 32; i++)
        for (int j = 31; j < grad.cols - 32; j++)
            if (grad.at<uchar>(i, j) == 255) {
                crop.x = j - 31;
                crop.y = i - 31;
                // скормить mat(crop) сетке
            }

    // === раздуваем пиксели до диаметра kernelSize
    //int kernelSize = 5;
    //Mat kernel = getStructuringElement(MORPH_ELLIPSE, { kernelSize, kernelSize });
    
    //dilate(grad, grad, kernel);
    
    
    // === пытаемся отфильтровать "шум" раскрытием и закрытием областей
    //morphologyEx(grad, grad, MORPH_OPEN, kernel);
    //morphologyEx(grad, grad, MORPH_CLOSE, kernel);
    /*
    // === отсеиваем мелкие контуры
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    double minSize = 1000;
    findContours(grad, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
    grad = Mat::zeros(grad.size(), CV_8U); // перерисовОчка
    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) > minSize) {
            drawContours(grad, contours, (int)i, {255}, 1, cv::LINE_8, hierarchy, 0);
        }
    }

    // === пытаемся замкнуть оставшиеся контуры 
    dilate(grad, grad, kernel);
    kernelSize = 10;
    kernel.release();
    kernel = getStructuringElement(MORPH_ELLIPSE, { kernelSize, kernelSize });
    morphologyEx(grad, grad, MORPH_CLOSE, kernel);

    // === заливаем пустоты в контурах
    grad = extra::imfill(grad);
    */


    gradX.release();
    gradY.release();
    absGradX.release();
    absGradY.release();
    filteredX.release();
    filteredY.release();
    blured.release();
    mat.release();

    mat = grad;

    cout << "Processed in " << timer.getElapsedTime().asMilliseconds() << "ms\n";
}

int main()
{
    using namespace std;

    const int WIN_WIDTH = 1200;
    const int WIN_HEIGHT = 700;

    const int LOW_TRESHOLD = 20;
    const int HIGH_TRESHOLD = 70;

    vector<string> imagePath = {};

    extra::loadFilenames(DATASET_PATH, ".bmp", imagePath);
    //for (auto& it : imagePath)
    //    cout << it << '\n';

    if (imagePath.empty()) {
        cout << "Directory \"" << DATASET_PATH << "\" is empty or doesn't exist.\n";
        return 0;
    }

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
    processGrayMat(grayMat);
    //cv::imwrite(imagePath[0] + ".jpg", grayMat);

    extra::cvtGrayMatToImage(grayMat, image);
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
                processGrayMat(grayMat);
                //cv::imwrite(imagePath[counter] + ".jpg", grayMat);

                extra::cvtGrayMatToImage(grayMat, image);
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