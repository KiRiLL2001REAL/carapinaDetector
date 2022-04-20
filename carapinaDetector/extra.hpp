#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <SFML/Graphics.hpp>
#include <opencv2/opencv.hpp>

namespace extra {
	using namespace std;

	// Получение пути файлов с заданным расширением в вектор из указанного каталога, включая подкаталоги
	void loadFilenames(const string& folder, const string& extension, vector<string>& out);
	// конвертация gray cv::Mat в sf::Image
	void cvtGrayMatToImage(const cv::Mat& mat, sf::Image& image);
	// конвертация rgb cv::Mat в sf::Image
	void cvtRGBMatToImage(const cv::Mat& mat, sf::Image& image);

	cv::Mat imfill(const cv::Mat& mat);

	// нарисовать оси направления (используется в getOrientation)
	void drawAxis(cv::Mat& img, cv::Point p, cv::Point q, cv::Scalar colour, const float scale = 0.2);
	// получить угол наклона контура в радианах + нарисовать центр и оси направления
	double getOrientation(const vector<cv::Point>& pts, cv::Mat& img);

	double lengthVector(cv::Point2d p);

	cv::Point2d normalize(cv::Point2d p);
}

void extra::loadFilenames(const string& folder, const string& extension, vector<string>& out)
{
	if (!filesystem::exists(folder))
		return;
	for (auto& it : filesystem::directory_iterator(folder)) {
		if (it.is_directory()) {
			loadFilenames(it.path().string(), extension, out);
			continue;
		}
		auto path = it.path();
		if (path.extension() == extension)
			out.push_back(path.string());
	}
}

void extra::cvtGrayMatToImage(const cv::Mat& mat, sf::Image& image)
{
	cv::Mat tmpMat;
	cv::cvtColor(mat, tmpMat, cv::COLOR_GRAY2RGBA);
	image.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
	tmpMat.release();
}

void extra::cvtRGBMatToImage(const cv::Mat& mat, sf::Image& image)
{
	cv::Mat tmpMat;
	cv::cvtColor(mat, tmpMat, cv::COLOR_RGB2RGBA);
	image.create(tmpMat.cols, tmpMat.rows, tmpMat.ptr());
	tmpMat.release();
}

cv::Mat extra::imfill(const cv::Mat& mat)
{
	using namespace cv;

	Mat filled = mat.clone();
	
	// === расширяем матрицу, чтобы гарантированно не попасть в закрашенный пиксель при заливке
	Mat additionalColumns = Mat::zeros(filled.rows, 2, filled.type());
	Mat additionalRows = Mat::zeros(2, filled.cols + 2, filled.type());
	hconcat(filled, additionalColumns, filled);
	vconcat(filled, additionalRows, filled);

	// === смещаем матрицу на 1 ячейку вправо-вниз
	Mat translationMat = (Mat_<double>(2, 3) << 
		1, 0, 1, 
		0, 1, 1);
	warpAffine(filled, filled, translationMat, filled.size());

	// === делаем нужыне вещи
	floodFill(filled, Point(0, 0), Scalar(255));

	Mat invFilled;
	bitwise_not(filled, invFilled);
	filled.release();

	// размеры invFilled отличаются от размеров mat, поэтому выделяем область
	Rect crop = Rect(1, 1, mat.cols, mat.rows);
	Mat out = invFilled(crop) | mat;
	invFilled.release();

	return out;
}

void extra::drawAxis(cv::Mat& img, cv::Point p, cv::Point q, cv::Scalar colour, const float scale)
{
	using namespace cv;

	double angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	double hypotenuse = sqrt((double)
		((long long)p.y - q.y) * ((long long)p.y - q.y) +
		((long long)p.x - q.x) * ((long long)p.x - q.x));

	// Here we lengthen the arrow by a factor of scale
	q.x = (int)((long long)p.x - scale * hypotenuse * cos(angle));
	q.y = (int)((long long)p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 1, LINE_AA);
	// create the arrow hooks
	p.x = (int)((long long)q.x + 9 * cos(angle + CV_PI / 4));
	p.y = (int)((long long)q.y + 9 * sin(angle + CV_PI / 4));
	line(img, p, q, colour, 1, LINE_AA);
	p.x = (int)((long long)q.x + 9 * cos(angle - CV_PI / 4));
	p.y = (int)((long long)q.y + 9 * sin(angle - CV_PI / 4));
	line(img, p, q, colour, 1, LINE_AA);
}

double extra::getOrientation(const vector<cv::Point>& pts, cv::Mat& img)
{
	using namespace cv;

	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
	//Store the center of the object
	Point center = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}
	// Draw the principal components
	//circle(img, center, 3, Scalar(255, 255, 255), 2);
	Point p1 = center + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
	Point p2 = center - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
	drawAxis(img, center, p1, Scalar(0, 255, 0), 1);
	drawAxis(img, center, p2, Scalar(255, 255, 0), 5);
	double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
	return angle;
}

double extra::lengthVector(cv::Point2d p)
{
	return sqrt(p.x * p.x + p.y * p.y);
}

cv::Point2d extra::normalize(cv::Point2d p)
{
	double len = lengthVector(p);
	double inv_len = (1.0 / len);
	p.x *= inv_len;
	p.y *= inv_len;
	return p;
}
