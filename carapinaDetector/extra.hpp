#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <SFML/Graphics.hpp>
#include <opencv2/opencv.hpp>

namespace extra {
	using namespace std;

	// ��������� ���� ������ � �������� ����������� � ������ �� ���������� ��������, ������� �����������
	void loadFilenames(const string& folder, const string& extension, vector<string>& out);
	// ����������� gray cv::Mat � sf::Image
	void cvtGrayMatToImage(const cv::Mat& mat, sf::Image& image);
	// ����������� rgb cv::Mat � sf::Image
	void cvtRGBMatToImage(const cv::Mat& mat, sf::Image& image);

	cv::Mat imfill(const cv::Mat& mat);
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
	
	// === ��������� �������, ����� �������������� �� ������� � ����������� ������� ��� �������
	Mat additionalColumns = Mat::zeros(filled.rows, 2, filled.type());
	Mat additionalRows = Mat::zeros(2, filled.cols + 2, filled.type());
	hconcat(filled, additionalColumns, filled);
	vconcat(filled, additionalRows, filled);

	// === ������� ������� �� 1 ������ ������-����
	Mat translationMat = (Mat_<double>(2, 3) << 
		1, 0, 1, 
		0, 1, 1);
	warpAffine(filled, filled, translationMat, filled.size());

	// === ������ ������ ����
	floodFill(filled, Point(0, 0), Scalar(255));

	Mat invFilled;
	bitwise_not(filled, invFilled);
	filled.release();

	// ������� invFilled ���������� �� �������� mat, ������� �������� �������
	Rect crop = Rect(1, 1, mat.cols, mat.rows);
	Mat out = invFilled(crop) | mat;
	invFilled.release();

	return out;
}
