#include <iostream>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>

#include "extra.hpp"
#include "sift.h"


//const std::string DATASET_PATH = "C:\\Dataset";
const std::string DATASET_PATH = "C:\\hack";

void makeLopatkaMask(const cv::Mat& binarizedFlatMat, cv::Mat& dst, const std::string& path);

#define null_rep -1
#define component_cnt_max 100
#define delta_bound 2
struct deltStruct {
    int pix_num_start;
    int pix_num_finish;
    int delta;
};
int comparator(const void* x, const void* y) { return ((deltStruct*)x)->delta - ((deltStruct*)y)->delta; }
int getRep(int* rep_arr, int x) {
    if (rep_arr[x] == null_rep)
        return x;
    rep_arr[x] = getRep(rep_arr, rep_arr[x]);
    return rep_arr[x];
}
void templatePictureMaker(const cv::Mat& src, cv::Mat& flat, cv::Mat& binarizedFlat, int start_i, int start_j, int ni, int nj);

int main()
{
    using namespace std;

    const int WIN_WIDTH = 1200;
    const int WIN_HEIGHT = 700;

    vector<string> imagePath = {};

    extra::loadFilenames(DATASET_PATH, ".jpg", imagePath);
    extra::loadFilenames(DATASET_PATH, ".bmp", imagePath);

    
    map<int, cv::Mat> descriptors;
    loadModels("C:\\Repo\\carapinaDetector\\descriptors\\", descriptors);

    if (imagePath.empty()) {
        cout << "Directory \"" << DATASET_PATH << "\" is empty or doesn't exist.\n";
        return 0;
    }

    sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Carapina detector", sf::Style::Close);
    sf::Image image;
    sf::Texture texture;
    sf::Sprite sprite;
    sf::Font font;
    font.loadFromFile("C:\\Windows\\Fonts\\courbd.ttf");
    sf::Text text;
    text.setString(imagePath[5]);
    text.setFont(font);
    text.setFillColor(sf::Color::White);
    text.setOutlineThickness(1.5);
    text.setOutlineColor(sf::Color::Black);
    text.setCharacterSize(28);

    //cv::namedWindow("rescaledMat");

    cv::Mat grayMat = cv::imread(imagePath[5], cv::IMREAD_GRAYSCALE);
    cv::Mat binarizedFlatMat;
    cv::Mat flatMat;

    cv::Mat tmpMat;
    cv::cvtColor(grayMat, tmpMat, cv::COLOR_GRAY2RGBA);
    image.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
    tmpMat.release();
    texture.loadFromImage(image);
    sprite.setTexture(texture);

    float scale = float(WIN_WIDTH) / sprite.getLocalBounds().width;
    sprite.setScale({ scale, scale });

    int counter = 5;
    const int imgDelayMillis = 0;

    bool finished = false;
    bool needUpdate = true;
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
                switch (e.key.code) {
                case sf::Keyboard::Escape:
                    window.close();
                    break;
                case sf::Keyboard::Right:
                    needUpdate = true;
                    counter++;
                    if (counter >= imagePath.size())
                        counter = 0;
                    break;
                case sf::Keyboard::Left:
                    needUpdate = true;
                    counter--;
                    if (counter < 0)
                        counter = (int)imagePath.size() - 1;
                    break;
                }
            }
        }

        if (needUpdate) {
            needUpdate = false;

            grayMat.release();
            grayMat = cv::imread(imagePath[counter], cv::IMREAD_GRAYSCALE);

            int img_name_ptr = imagePath[counter].find_last_of("\\");
            int img_extension_ptr = imagePath[counter].find_last_of(".");
            string imgName = imagePath[counter].substr(img_name_ptr + 1, img_extension_ptr - img_name_ptr - 1);

            // background - 0
            // lopatka - 1
            // sled_ot_frezi - 2 <-
            // zaboina - 3       <-
            // carapina - 4
            // riska - 5
            // nadir - 6         <-
            // chernota - 7      <-

            map<int, cv::Mat> masks;
            GetMasks(grayMat, descriptors, 6, masks);

            binarizedFlatMat.release();
            flatMat.release();
            templatePictureMaker(grayMat, flatMat, binarizedFlatMat, 0, 0, grayMat.rows, grayMat.cols);
            makeLopatkaMask(binarizedFlatMat, masks[1], imagePath[counter]);

            cv::Mat outMASK = cv::Mat::zeros(grayMat.size(), CV_8U);
            for (auto& maskIt : masks) {
                for (int i = 0; i < outMASK.rows; i++)
                    for (int j = 0; j < outMASK.cols; j++)
                        if (maskIt.second.at<uchar>(i, j) == 255)
                            outMASK.at<uchar>(i, j) = maskIt.first * 30;
            }

            imwrite("C:\\out\\" + imgName + "_seg.png", outMASK);

            outMASK.release();
            for (auto& it : masks)
                it.second.release();
            map<int, cv::Mat>().swap(masks);

            cv::cvtColor(grayMat, tmpMat, cv::COLOR_GRAY2RGBA);
            //for (int i = 0; i < grayMat.rows; i++)
            //    for (int j = 0; j < grayMat.cols; j++) {
            //        if (lopatkaMask.at<uchar>(i, j) == 255) {
            //            tmpMat.at<cv::Vec4b>(i, j)[0] = 255;
            //            tmpMat.at<cv::Vec4b>(i, j)[3] = 255;
            //        }
            //    }
            image.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
            tmpMat.release();
            texture.loadFromImage(image);

            text.setString(imagePath[counter]);
            cout << "Showed file \"" << imagePath[counter] << "\"\n";
        }

        window.clear();

        sprite.setTexture(texture);
        window.draw(sprite);

        window.draw(text);
        window.display();

        this_thread::sleep_for(chrono::milliseconds(1));
    }

    cv::destroyAllWindows();

    return 0;
}

void makeLopatkaMask(const cv::Mat& binarizedFlatMat, cv::Mat& dst, const std::string& path) {
    using namespace cv;
    using namespace std;

    if (!dst.empty())
        dst.release();

    Mat bwFlat = binarizedFlatMat.clone();

    Mat dilated;
    Mat dk11 = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
    Mat dk15 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    Mat ek3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(bwFlat, dilated, dk11);
    erode(dilated, dilated, ek3);
    dilate(dilated, dilated, dk15);
    bwFlat.release();

    Mat cannyed;
    Canny(dilated, cannyed, 10, 255, 3);
    dilated.release();

    Mat dk3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(cannyed, cannyed, dk3);
    dilate(cannyed, cannyed, dk3);
    erode(cannyed, cannyed, dk3);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(cannyed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    cannyed.release();
    Rect arect = boundingRect(contours[0]);

    size_t maxIndexSize = 0;
    size_t prevMaxIndexSize = 0;
    double maxSize = max(arect.size().width, arect.size().height);
    double prevMaxSize = maxSize;

    size_t maxIndexArea = 0;
    size_t prevMaxIndexArea = 0;
    double maxArea = arect.area();
    double prevMaxArea = maxArea;

    for (size_t i = 1; i < contours.size(); i++) {
        arect = boundingRect(contours[i]);
        double size = max(arect.size().width, arect.size().height);
        if (size > maxSize) {
            prevMaxSize = maxSize;
            prevMaxIndexSize = maxIndexSize;
            maxSize = size;
            maxIndexSize = i;
        }
        else if (size > prevMaxSize) {
            prevMaxSize = size;
            prevMaxIndexSize = i;
        }
        double area = arect.area();
        if (area > maxArea) {
            prevMaxArea = maxArea;
            prevMaxIndexArea = maxIndexArea;
            maxArea = area;
            maxIndexArea = i;
        }
        else if (area > prevMaxArea) {
            prevMaxArea = area;
            prevMaxIndexArea = i;
        }
    }
    vector<Point> contour = vector<Point>(contours[maxIndexSize].begin(), contours[maxIndexSize].end());
    contour.insert(contour.end(), contours[prevMaxIndexSize].begin(),     contours[prevMaxIndexSize].end());
    contour.insert(contour.end(), contours[maxIndexArea].begin(),         contours[maxIndexArea].end());
    contour.insert(contour.end(), contours[prevMaxIndexArea].begin(),     contours[prevMaxIndexArea].end());

    RotatedRect rrect = minAreaRect(contour);
    vector<Point2f> rectPoints(4);
    rrect.points(rectPoints.data());

    vector<vector<Point>> rectContour(1);
    rectContour[0] = vector<Point>(4);
    for (int i = 0; i < 4; i++)
        rectContour[0][i] = Point(rectPoints[i].x, rectPoints[i].y);

    Mat mask = Mat::zeros(binarizedFlatMat.size(), CV_8UC1);
    drawContours(mask, rectContour, 0, Scalar(255), 1);
    floodFill(mask, Point(rrect.center), Scalar(255));

    dst = mask;
}

void templatePictureMaker(const cv::Mat& src, cv::Mat& flat, cv::Mat& binarizedFlat, int start_i, int start_j, int ni, int nj)
{
    using namespace cv;

    Mat blured;
    GaussianBlur(src, blured, Size(3, 3), 0);
    /*namedWindow("Source Image", WINDOW_NORMAL);
    imshow("Source Image", src);*/

    // Вместо координат вершин будем хранить их номер от начала ядра. Пиксель с номером 0 имеет координаты [0;0],
    // с номером 1 - [0;1], с номером n - [1; 0].
    int delt_struct_size = 2 * ni * nj;

    deltStruct* delt_arr = new deltStruct[delt_struct_size];
    int arr_ptr = 0; //min_color = (int)src.at<uchar>(start_i, start_j);

    for (int i = start_i; i < start_i + ni - 1; i++)
    {
        for (int j = start_j; j < start_j + nj - 1; j++)
        {
            /*if ((int)src.at<uchar>(i, j) < min_color)
              min_color = (int)src.at<uchar>(i, j);*/

            delt_arr[arr_ptr].pix_num_start = (i - start_i) * nj + (j - start_j);
            delt_arr[arr_ptr].pix_num_finish = (i - start_i + 1) * nj + (j - start_j); // Вниз
            delt_arr[arr_ptr].delta = abs((int)blured.at<uchar>(i, j) - (int)blured.at<uchar>(i + 1, j));
            arr_ptr++;

            delt_arr[arr_ptr].pix_num_start = (i - start_i) * nj + (j - start_j);
            delt_arr[arr_ptr].pix_num_finish = (i - start_i) * nj + (j - start_j) + 1; // Вправо
            delt_arr[arr_ptr].delta = abs((int)blured.at<uchar>(i, j) - (int)blured.at<uchar>(i, j + 1));
            arr_ptr++;
        }
    }

    // Обрабатываем крайние пиксели отдельно (последняя строка и последний столбец)

    for (int j = start_j; j < start_j + nj - 1; j++)
    {
        /*if ((int)src.at<uchar>(start_i + ni - 1, j) < min_color)
          min_color = (int)src.at<uchar>(start_i + ni - 1, j);*/

        delt_arr[arr_ptr].pix_num_start = nj * (ni - 1) + (j - start_j);
        delt_arr[arr_ptr].pix_num_finish = nj * (ni - 1) + (j - start_j) + 1; // Вправо
        delt_arr[arr_ptr].delta = abs((int)blured.at<uchar>(start_i + ni - 1, j) - (int)blured.at<uchar>(start_i + ni - 1, j + 1));
        arr_ptr++;
    }

    for (int i = start_i; i < start_i + ni - 1; i++)
    {
        /*if ((int)src.at<uchar>(i, start_j + nj - 1) < min_color)
          min_color = (int)src.at<uchar>(i, start_j + nj - 1);*/

        delt_arr[arr_ptr].pix_num_start = (i - start_i) * nj + nj - 1;
        delt_arr[arr_ptr].pix_num_finish = (i - start_i + 1) * nj + nj - 1; // Вниз
        delt_arr[arr_ptr].delta = abs((int)blured.at<uchar>(i, start_j + nj - 1) - (int)blured.at<uchar>(i + 1, start_j + nj - 1));
        arr_ptr++;
    }

    qsort(delt_arr, arr_ptr, sizeof(deltStruct), comparator); // Сортируем по возрастанию delta

    /*for (int i = 0; i < arr_ptr; i++)
      cout << delt_arr[i].delta << endl;*/

    int* rep_arr = new int[ni * nj]; // непересекающееся множество
    std::fill_n(rep_arr, ni * nj, null_rep);

    // Указатель на дельту. Хранит номер первого элемента в массиве delt_arr с delta = i. 
    // Количество таких элементов: delta_ptr[i + 1] - delta_ptr[i]
    int* delta_ptr = new int[delt_arr[arr_ptr - 1].delta + 2];
    std::fill_n(delta_ptr, delt_arr[arr_ptr - 1].delta + 2, -1);
    //int* comp_cnt = new int[component_cnt_max] {0};

    for (int i = 0; i < arr_ptr; i++)
        if (delta_ptr[delt_arr[i].delta] == -1) // Встречен первый элемент с такой delta
            delta_ptr[delt_arr[i].delta] = i;
    delta_ptr[delt_arr[arr_ptr - 1].delta + 1] = arr_ptr;

    //Убираем -1, если они остались
    for (int i = 0, j = 0; i < delt_arr[arr_ptr - 1].delta + 1; i++)
    {
        if (delta_ptr[i] == -1) // Не было элементов с такой delta
        {
            j = i + 1;
            while (delta_ptr[j] == -1)
                j++;
            delta_ptr[i] = delta_ptr[j];
        }
    }

    //for (int i = 0; i < delt_arr[arr_ptr - 1].delta + 2; i++)
    //  cout << delta_ptr[i] << endl;

    int cur_delta = 1, u, v;
    int component_cnt = (ni - start_i) * (nj - start_j);

    //component_cnt += (delta_ptr[cur_delta] - delta_ptr[cur_delta - 1]);

    for (int i = delta_ptr[cur_delta - 1]; i < delta_ptr[delta_bound]; i++)
    {
        // Не находятся в одном множестве
        u = getRep(rep_arr, delt_arr[i].pix_num_start);
        v = getRep(rep_arr, delt_arr[i].pix_num_finish);

        if (u != v)
        {
            if (rand() % 2 == 0)
                rep_arr[u] = v;
            else
                rep_arr[v] = u;
            //component_cnt--;
        }
    }
    // Подсчитываем количество вершин в каждой компоненте и сумму цветов
    int* color_sum = new int[ni * nj]{ 0 };
    int* vertex_cnt = new int[ni * nj]{ 0 };
    for (int i = 0; i < ni * nj; i++)
    {
        u = getRep(rep_arr, i);
        color_sum[u] += (int)blured.at<uchar>(start_i + u / nj, start_j + u % nj);
        vertex_cnt[u]++;
    }

    Mat resImg = Mat::zeros(ni, nj, CV_8UC1);
    Mat maskImg = Mat::zeros(ni, nj, CV_8UC1);
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++)
        {
            u = getRep(rep_arr, i * nj + j);
            resImg.at<uchar>(i, j) = floor((float)color_sum[u] / (vertex_cnt[u]));// - min_color;

            uchar color;
            if (vertex_cnt[u] <= 25)
                color = 255;
            else color = 0;
            maskImg.at<uchar>(i, j) = color;
        }

    delete[] delt_arr;
    delete[] rep_arr;
    delete[] delta_ptr;
    delete[] color_sum;
    delete[] vertex_cnt;

    flat = resImg;
    binarizedFlat = maskImg;
}