#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <map>

using namespace std;
using namespace cv;
namespace fs = filesystem;

void loadDescriptors(string path, Mat& descriptor) {
	ifstream File(path, ios::binary | ios::in);
	int rows, cols, type;
	File.read((char*)&rows, sizeof(int));
	File.read((char*)&cols, sizeof(int));
	File.read((char*)&type, sizeof(int));

	descriptor = Mat::zeros(rows, cols, type);

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			File.read((char*)&descriptor.at<float>(row, col), sizeof(float));
		}
	}
	File.close();
}

void loadModels(string path, map<int, Mat>& descriptors) {
	for (auto& it : fs::directory_iterator(path)) {
		string name = it.path().string().substr(it.path().string().rfind("\\") + 1);
		string rassh = name.substr(name.size() - 3);
		name.erase(name.size() - 4);
		if (it.is_directory() || rassh != "dsc")
			continue;
		loadDescriptors(it.path().string(), descriptors[std::stoi(name)]);
	}
}

double kv(Mat& src1, Mat& src2, int row1, int row2) {
	double k = 0;
	double loc = 0;
	for (int col = 0; col < src1.cols; col++)
	{
		loc = src1.at<float>(row1, col) - src2.at<float>(row2, col);
		k += loc * loc;
	}
	return k;
}

float kv(Mat& src1, Mat& src2, int& row1, int& row2, float& minK) {
	double k = 0;
	double loc = 0;
	for (int col = 0; col < src1.cols; col++)
	{
		loc = src1.at<float>(row1, col) - src2.at<float>(row2, col);
		k += loc * loc;
		if (col % 3 && k > minK)
			return k;
	}
	return k;
}

void match(Mat& src1, Mat& src2, vector<DMatch>& matches) {
	matches.clear();
	for (int row1 = 0; row1 < src1.rows; row1++) {
		DMatch q;
		q.queryIdx = row1;
		q.distance = kv(src1, src2, row1, 0);
		q.trainIdx = 0;
		double rast;
		for (int row2 = 1; row2 < src2.rows; row2++) {
			rast = kv(src1, src2, row1, row2, q.distance);
			if (rast < q.distance)
			{
				q.distance = rast;
				q.trainIdx = row2;
				if (q.distance == 0)
					break;
			}
		}
		matches.push_back(q);
	}
}

//*************************************************************************TODO распараллелить
vector<int> recognize(map<int, Mat>& descriptors, Mat Desc) {
	vector<int> qm;
	map<int, vector<DMatch>> full;
	for (auto& it : descriptors)
	{
		if (it.second.rows > 0)
			//BM->match(Desc, descriptors[it.first], full[it.first]);
			match(Desc, descriptors[it.first], full[it.first]);
		//full[it.first] = match;
	}

	if (full.size() > 0)
	{
		auto it = full.begin();
		for (int i = 0; i < it->second.size(); i++) {
			qm.push_back(it->first);
		}
	}
	else qm.resize(Desc.rows, -2);

	for (auto& it : full) {
		for (int i = 0; i < qm.size(); i++) {
			if (full[qm[i]][i].distance > it.second[i].distance)
				qm[i] = it.first;
		}
	}
	return qm;
}

void GetMasks(Mat& src, map<int, Mat>& descriptors, double SizeThreshold, map<int, Mat>& Masks) {
	Mat mySrc = src.clone();

	vector<KeyPoint> keypoints;
	Mat desctriptor;

	Ptr<SIFT> siftPtr = SIFT::create(0, 3, 0.04, 5);

	siftPtr->detectAndCompute(mySrc, Mat(), keypoints, desctriptor);

	vector<int> minq = recognize(descriptors, desctriptor);

	map<int, vector<KeyPoint>> Dkeypoints;

	for (int i = 0; i < keypoints.size(); i++) {
		if (minq[i] > -1 && keypoints[i].size > SizeThreshold)
			Dkeypoints[minq[i]].push_back(keypoints[i]);
	}

	for (auto it : Dkeypoints) {
		Masks[it.first] = Mat::zeros(mySrc.size(), CV_8UC1);
		for (int i = 0; i < it.second.size(); i++) {
			cv::circle(Masks[it.first], Point(Dkeypoints[it.first][i].pt.x, Dkeypoints[it.first][i].pt.y), Dkeypoints[it.first][i].size / 2, Scalar::all(255), -1);
			Dkeypoints[it.first][i].size *= 2;
		}
	}
}