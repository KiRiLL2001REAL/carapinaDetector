#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <SFML/Graphics.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>


using namespace std;


namespace extra {

	// Получение пути файлов с заданным расширением в вектор из указанного каталога, включая подкаталоги
	void loadFilenames(const string& folder, const string& extension, vector<string>& out);
	// конвертация gray cv::Mat в sf::Image
	void cvtGrayMatToImage(const cv::Mat& mat, sf::Image& image);
	// конвертация rgb cv::Mat в sf::Image
	void cvtRGBMatToImage(const cv::Mat& mat, sf::Image& image);
}

void extra::loadFilenames(const string& folder, const string& extension, vector<string>& out)
{
	if (!filesystem::exists(folder))
		return;
	for (auto& it : filesystem::directory_iterator(folder)) {
		if (it.is_directory()) {
			//loadFilenames(it.path().string(), extension, out);
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

long long srkvotkl(vector<vector<int>>& src, vector<vector<int>>& dst, int& threshold) {
	long long res = 0;
	int loc;
	for (int i = 0; i < dst.size(); i++) {
		for (int j = 0; j < dst[i].size(); j++) {
			loc = dst[i][j] - src[i][j];
			res += (long long)loc * loc;
		}
		if (threshold <= res)
			return -1;
	}
	return res;
}

int getMinOtkl(vector<vector<int>>& src, vector<vector<vector<int>>>dst, int& threshold) {
	long long minsqr = srkvotkl(src, dst[0], threshold);
	for (int i = 1; i < dst.size(); i++) {
		long long s = srkvotkl(src, dst[i], threshold);
		if ((s < minsqr || minsqr == -1) && s >= 0) {
			minsqr = s;
		}
	}
	if (minsqr == -1)
		return minsqr;
	return minsqr / 1089;
}

void loadFromFile(string filename, vector<vector<vector<int>>>& result, int& threshold) {
	using namespace filesystem;
	ifstream File(filename);

	int size;
	File >> size >> threshold;
	for (int k = 0; k < size; k++) {
		vector<vector<int>> dst(33, vector<int>(33, 0));
		for (int i = 0; i < 33; i++) {
			for (int j = 0; j < 33; j++) {
				File >> dst[i][j];
			}
		}
		result.push_back(dst);
	}

	File.close();
}

void saveToFile(string filename, vector<vector<vector<int>>>& result, int threshold) {
	using namespace filesystem;
	ofstream File(filename);

	int size = result.size();
	File << size << " " << threshold << "\n";
	for (int k = 0; k < size; k++) {
		for (int i = 0; i < 33; i++) {
			for (int j = 0; j < 33; j++) {
				File << result[k][i][j] << " ";
			}
			File << "\n";
		}
	}

	File.close();
}