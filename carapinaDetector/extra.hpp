#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <SFML/Graphics.hpp>
#include <opencv2/opencv.hpp>
#include <exception>

namespace extra {
	using namespace std;

	// Получение пути файлов с заданным расширением в вектор из указанного каталога, включая подкаталоги
	void loadFilenames(const string& folder, const string& extension, vector<string>& out);
	// конвертация gray cv::Mat в sf::Image
	void cvtGrayMatToImage(const cv::Mat& mat, sf::Image& image);
	// конвертация rgb cv::Mat в sf::Image
	void cvtRGBMatToImage(const cv::Mat& mat, sf::Image& image);

	cv::Mat imfill(const cv::Mat& mat);

	void filterStrong(const cv::Mat& mat, cv::Mat& dst, char orientation);
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

void extra::filterStrong(const cv::Mat& mat, cv::Mat& dst, char orientation)
{
	using namespace cv;

	if (orientation != 'x' && orientation != 'y')
		throw std::exception("Unknown orientation");

	Mat _src = mat.clone();
	if (orientation == 'y')
		_src = _src.t();

	cv::Mat _dst = Mat::zeros(_src.size(), _src.type());

	for (int i = 0; i < _src.rows; i++) {
		if (_src.at<uchar>(i, 0) > _src.at<uchar>(i, 1)) {
			_dst.at<uchar>(i, 0) = _src.at<uchar>(i, 0);
		}
		for (int j = 1; j < _src.cols - 1; j++) {
			if (_src.at<uchar>(i, j) > _src.at<uchar>(i, j + 1) && \
				_src.at<uchar>(i, j) > _src.at<uchar>(i, j - 1))
			{
				_dst.at<uchar>(i, j) = _src.at<uchar>(i, j);
			}
		}
		if (_src.at<uchar>(i, _src.cols - 1) > _src.at<uchar>(i, _src.cols - 2)) {
			_dst.at<uchar>(i, _src.cols - 1) = _src.at<uchar>(i, _src.cols - 1);
		}
	}
	_src.release();

	if (orientation == 'y')
		_dst = _dst.t();

	dst.release();
	dst = _dst;
}
