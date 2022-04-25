#pragma once

#include <opencv2/Core.hpp>
#include <string>
#include <vector>
#include <random>
#include "json.hpp"

namespace cnn
{
	struct TensorSize;
	class Tensor;
	class ConvLayer;
	class MaxPoolingLayer;
	class ReLULayer;
	class SigmoidLayer;
	class Matrix;
	class FullyConnectedLayer;
}



// ����������� �������
struct cnn::TensorSize
{
	int depth; // �������
	int height; // ������
	int width; // ������

	TensorSize() {
		width = height = depth = 1;
	}
	TensorSize(int width, int height, int depth) {
		this->width = width;
		this->height = height;
		this->depth = depth;
	}
};



// ������
class cnn::Tensor
{
private:
	// ����������� �������
	TensorSize size;
	// �������� �������
	std::vector<double> values;
	// ������������ ������� �� ������ ��� ����������
	int dw;

	// ������������� �� ��������
	void init(int width, int height, int depth) {
		size.width = width; // ���������� ������
		size.height = height; // ���������� ������
		size.depth = depth; // ���������� �������

		dw = depth * width; // ���������� ������������ ������� �� ������ ��� ����������

		values = std::vector<double>((size_t)width * height * depth, 0); // ������ ������ �� width * height * depth �����
	}

public:
	// �����������-��������.
	Tensor() : Tensor(1, 1, 1) {}
	// �������� �� ��������
	Tensor(int width, int height, int depth) {
		init(width, height, depth);
	}
	// �������� �� �������
	Tensor(const TensorSize& size) {
		init(size.width, size.height, size.depth);
	}
	// �������� �� json
	Tensor(nlohmann::json js) {
		int width = 0, height = 0, depth = 0;
		width = js["size"]["w"].get<int>();
		height = js["size"]["h"].get<int>();
		depth = js["size"]["d"].get<int>();
		init(width, height, depth);
		for (int d = 0; d < depth; d++) {
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					values[(size_t)i * dw + (size_t)j * depth + d] = js["values"][d][i][j].get<double>();
				}
			}
		}
	}

	// ����������
	double& operator()(int d, int i, int j) {
		return values[(size_t)i * dw + (size_t)j * size.depth + d];
	}
	// ����������
	double operator()(int d, int i, int j) const {
		return values[(size_t)i * dw + (size_t)j * size.depth + d];
	}
	// ����������
	double& operator[](int i) {
		return values[i];
	}
	// ����������
	double operator[](int i) const {
		return values[i];
	}
	// ��������� �������
	TensorSize getSize() const {
		return size;
	}
	// ����� �������
	friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
		for (int d = 0; d < tensor.size.depth; d++) {
			for (int i = 0; i < tensor.size.height; i++) {
				for (int j = 0; j < tensor.size.width; j++) {
					os << tensor.values[(size_t)i * tensor.dw + (size_t)j * tensor.size.depth + d] << " ";
				}
				os << std::endl;
			}
			os << std::endl;
		}
		return os;
	}
	// ��������� ��������
	void setValue(int d, int i, int j, double value) {
		values[(size_t)i * dw + (size_t)j * size.depth + d] = value;
	}
	// ����� ������� � json
	nlohmann::json getJson() {
		nlohmann::json js;
		js["size"]["w"] = size.width;
		js["size"]["h"] = size.height;
		js["size"]["d"] = size.depth;
		for (int d = 0; d < size.depth; d++) {
			for (int i = 0; i < size.height; i++) {
				for (int j = 0; j < size.width; j++) {
					js["values"][d][i][j] = values[(size_t)i * dw + (size_t)j * size.depth + d];
				}
			}
		}
		return js;
	}
};



// ���� ������
class cnn::ConvLayer
{
private:
	// ��������� ��������� �����
	std::default_random_engine generator;
	// � ���������� ��������������
	std::normal_distribution<double> distribution;
	// ������ �����
	TensorSize inputSize;
	// ������ ������
	TensorSize outputSize;
	// �������
	std::vector<Tensor> W;
	// ��������
	std::vector<double> b;
	// ��������� ��������
	std::vector<Tensor> dW;
	// ��������� ��������
	std::vector<double> db;
	// ���������� ������
	int P;
	// ��� ������
	int S;
	// ���������� ��������
	int fc;
	// ������ ��������
	int fs;
	// ������� ��������
	int fd;

	// ������������� ������� �������������
	void initWeights() {
		// ���������� �� ������� �� ��������
		for (int index = 0; index < fc; index++) {
			for (int i = 0; i < fs; i++) {
				for (int j = 0; j < fs; j++) {
					for (int k = 0; k < fd; k++) {
						W[index](k, i, j) = distribution(generator); // ���������� ��������� ����� � ���������� ��� � ������� �������
					}
				}
			}
			b[index] = 0.01; // ��� �������� ������������� � 0.01
		}
	};

public:
	/*
	�����������-��������.
	������������ ������ ��� ������������� ���������� ������� ���������
	�������   ����.  ��  ��������   �������  ����������   �����������.
	*/
	ConvLayer() : ConvLayer(TensorSize(1, 1, 1), 1, 1, 1, 1) {}
	// ����������� ���������� ����
	ConvLayer(TensorSize size, int fc, int fs, int P, int S) :
		distribution(0.0, sqrt(2.0 / ((size_t)fs * fs * size.depth)))
	{
		// ���������� ������� ������
		inputSize.width = size.width;
		inputSize.height = size.height;
		inputSize.depth = size.depth;

		// ��������� �������� ������
		outputSize.width = (size.width - fs + 2 * P) / S + 1;
		outputSize.height = (size.height - fs + 2 * P) / S + 1;
		outputSize.depth = fc;

		this->P = P; // ��������� ���������� ������
		this->S = S; // ��������� ��� ������

		this->fc = fc; // ��������� ����� ��������
		this->fs = fs; // ��������� ������ ��������
		this->fd = size.depth; // ��������� ������� ��������

		// ��������� fc �������� ��� ����� �������� � �� ����������
		W = std::vector<Tensor>(fc, Tensor(fs, fs, fd));
		dW = std::vector<Tensor>(fc, Tensor(fs, fs, fd));

		// ��������� fc ����� ��� ����� �������� � �� ����������
		b = std::vector<double>(fc, 0);
		db = std::vector<double>(fc, 0);

		initWeights(); // �������������� ������� ������������
	}
	// ����������� ���������� ���� �� Json
	ConvLayer(nlohmann::json js) {
		// ������� ������
		inputSize.width = js["inputSize"]["w"].get<int>();
		inputSize.height = js["inputSize"]["h"].get<int>();
		inputSize.depth = js["inputSize"]["d"].get<int>();

		// �������� ������
		outputSize.width = js["outputSize"]["w"].get<int>();
		outputSize.height = js["outputSize"]["h"].get<int>();
		outputSize.depth = js["outputSize"]["d"].get<int>();

		P = js["P"].get<int>(); // ���������� ������
		S = js["S"].get<int>(); // ��� ������

		fc = js["fc"].get<int>(); // ����� ��������
		fs = js["fs"].get<int>(); // ������ ��������
		fd = js["fd"].get<int>(); // ������� ��������

		// fc �������� ����� �������� � �� ����������
		W = std::vector<Tensor>(fc);
		for (int i = 0; i < fc; i++)
			W[i] = Tensor(js["W"][i]);
		dW = std::vector<Tensor>(fc, Tensor(fs, fs, fd));

		// fc ����� ����� �������� � �� ����������
		b = std::vector<double>(fc);
		for (int i = 0; i < fc; i++)
			b[i] = js["b"][i].get<double>();
		db = std::vector<double>(fc, 0);
	}
	// ������ ��������������� �������
	Tensor forward(const Tensor& X) {
		Tensor output(outputSize); // ������ �������� ������
		// ���������� �� ������� �� ��������
		for (int f = 0; f < fc; f++) {
			for (int y = 0; y < outputSize.height; y++) {
				for (int x = 0; x < outputSize.width; x++) {
					double sum = b[f]; // ����� ���������� ��������
					// ���������� �� ��������
					for (int i = 0; i < fs; i++) {
						for (int j = 0; j < fs; j++) {
							int i0 = S * y + i - P;
							int j0 = S * x + j - P;
							// ��������� ��� ������ �������� ������� �������� �������, �� ������ ���������� ��
							if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
								continue;
							// ���������� �� ���� ������� ������� � ������� �����
							for (int c = 0; c < fd; c++)
								sum += X(c, i0, j0) * W[f](c, i, j);
						}
					}
					output(f, y, x) = sum; // ���������� ��������� ������ � �������� ������
				}
			}
		}

		return output; // ���������� �������� ������
	}
	// �������� ��������������� ������
	Tensor backward(const Tensor& dout, const Tensor& X) {
		TensorSize size; // ������ �����
		// ����������� ������ ��� �����
		size.height = S * (outputSize.height - 1) + 1;
		size.width = S * (outputSize.width - 1) + 1;
		size.depth = outputSize.depth;
		Tensor deltas(size); // ������ ������ ��� �����
		// ����������� �������� �����
		for (int d = 0; d < size.depth; d++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					deltas(d, i * S, j * S) = dout(d, i, j);
		// ����������� ��������� ����� �������� � ��������
		for (int f = 0; f < fc; f++)
			for (int y = 0; y < size.height; y++)
				for (int x = 0; x < size.width; x++) {
					double delta = deltas(f, y, x); // ���������� �������� ���������
					for (int i = 0; i < fs; i++) {
						for (int j = 0; j < fs; j++) {
							int i0 = i + y - P;
							int j0 = j + x - P;
							// ���������� ��������� �� ������� ��������
							if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
								continue;
							// ���������� �������� �������
							for (int c = 0; c < fd; c++)
								dW[f](c, i, j) += X(c, i0, j0) * delta;
						}
					}
					db[f] += delta; // ���������� �������� ��������
				}
		int pad = fs - 1 - P; // �������� �������� ����������
		Tensor dX(inputSize); // ������ ������ ���������� �� �����
		// ����������� �������� ���������
		for (int y = 0; y < inputSize.height; y++)
			for (int x = 0; x < inputSize.width; x++)
				for (int c = 0; c < fd; c++) {
					double sum = 0; // ����� ��� ���������
					// ��� �� ���� ������� ������������� ��������
					for (int i = 0; i < fs; i++) {
						for (int j = 0; j < fs; j++) {
							int i0 = y + i - pad;
							int j0 = x + j - pad;
							// ���������� ��������� �� ������� ��������
							if (i0 < 0 || i0 >= size.height || j0 < 0 || j0 >= size.width)
								continue;
							// ��������� �� ���� ��������
							for (int f = 0; f < fc; f++)
								sum += W[f](c, fs - 1 - i, fs - 1 - j) * deltas(f, i0, j0); // ��������� ������������ ��������� �������� �� ������
						}
					}
					dX(c, y, x) = sum; // ���������� ��������� � ������ ���������
				}
		return dX; // ���������� ������ ����������
	}
	// ���������� ������� �������������
	void updateWeights(double learning_rate) {
		for (int index = 0; index < fc; index++) {
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					for (int d = 0; d < fd; d++) {
						W[index](d, i, j) -= learning_rate * dW[index](d, i, j); // �������� ��������, ���������� �� �������� ��������
						dW[index](d, i, j) = 0; // �������� �������� �������
					}
			b[index] -= learning_rate * db[index]; // �������� ��������, ���������� �� �������� ��������
			db[index] = 0; // �������� �������� ���� ��������
		}
	}
	// ��������� ���� ������� �� �������
	void setWeight(int index, int d, int i, int j, double weight) {
		W[index](d, i, j) = weight;
	}
	// ��������� ���� �������� �� �������
	void setBias(int index, double bias) {
		b[index] = bias;
	}
	// ������ ��������� �������
	TensorSize getOutputSize() const {
		return outputSize;
	}
	std::vector<cv::Mat> getVisualRepresentation() {
		std::vector<cv::Mat> res = {};
		const int width = W[0].getSize().width;
		const int height = W[0].getSize().height;
		unsigned char* matr = new unsigned char[(size_t)width * height];
		for (int c = 0; c < fc; c++) {
			for (int i = 0; i < width; i++)
				for (int j = 0; j < height; j++) {
					int temp = (int)round(W[c](0, i, j) * 255);
					temp = temp < 0 ? 0 : (temp > 255 ? 255 : temp);
					matr[i * height + j] = (unsigned char)temp;
				}
			res.push_back(cv::Mat(height, width, CV_8UC1, matr));
		}
		delete[] matr;
		return res;
	}
	// ����� ���� � json
	nlohmann::json getJson() {
		nlohmann::json js;

		// ������� ������
		js["inputSize"]["w"] = inputSize.width;
		js["inputSize"]["h"] = inputSize.height;
		js["inputSize"]["d"] = inputSize.depth;

		// �������� ������
		js["outputSize"]["w"] = outputSize.width;
		js["outputSize"]["h"] = outputSize.height;
		js["outputSize"]["d"] = outputSize.depth;

		js["P"] = P; // ���������� ������
		js["S"] = S; // ��� ������

		js["fc"] = fc; // ����� ��������
		js["fs"] = fs; // ������ ��������
		js["fd"] = fd; // ������f ��������

		// fc �������� ����� �������� � �� ����������
		for (int i = 0; i < fc; i++)
			js["W"][i] = W[i].getJson();

		// fc ����� ����� �������� � �� ����������
		for (int i = 0; i < fc; i++)
			js["b"][i] = b[i];

		return js;
	}
};



// ���� max pooling'�
class cnn::MaxPoolingLayer
{
private:
	// ������ �����
	TensorSize inputSize;
	// ������ ������
	TensorSize outputSize;
	// �� ������� ��� ����������� �����������
	int scale;
	// ����� ��� ����������
	Tensor mask;

public:
	/*
	�����������-��������.
	������������ ������ ��� ������������� ���������� ������� ���������
	�������   ����.  ��  ��������   �������  ����������   �����������.
	*/
	MaxPoolingLayer() : MaxPoolingLayer(TensorSize(1, 1, 1)) {}
	// ����������� maxpool ����
	MaxPoolingLayer(TensorSize size, int scale = 2) :
		mask(size)
	{
		// ���������� ������� ������
		inputSize.width = size.width;
		inputSize.height = size.height;
		inputSize.depth = size.depth;

		// ��������� �������� ������
		outputSize.width = size.width / scale;
		outputSize.height = size.height / scale;
		outputSize.depth = size.depth;

		this->scale = scale; // ���������� ����������� ����������

		// ��������� ����� ���������
		TensorSize mask_size = mask.getSize();
		for (int d = 0; d < mask_size.depth; d++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					mask(d, i, j) = (double)rand() / RAND_MAX;
	}
	// ����������� maxpool ���� �� Json
	MaxPoolingLayer(nlohmann::json js) {
		// ������� ������
		inputSize.width = js["inputSize"]["w"].get<int>();
		inputSize.height = js["inputSize"]["h"].get<int>();
		inputSize.depth = js["inputSize"]["d"].get<int>();

		// �������� ������
		outputSize.width = js["outputSize"]["w"].get<int>();
		outputSize.height = js["outputSize"]["h"].get<int>();
		outputSize.depth = js["outputSize"]["d"].get<int>();

		scale = js["scale"].get<int>(); // ����������� ����������

		mask = Tensor(js["mask"]); // �����
	}
	// ������ ���������������
	Tensor forward(const Tensor& X) {
		Tensor output(outputSize); // ������ �������� ������
		// ���������� �� ������� �� �������
		for (int d = 0; d < inputSize.depth; d++)
			for (int i = 0; i < inputSize.height; i += scale)
				for (int j = 0; j < inputSize.width; j += scale) {
					int imax = i; // ������ ������ ���������
					int jmax = j; // ������ ������� ���������
					double max = X(d, i, j); // ��������� �������� ��������� - �������� ������ ������ ����������
					// ���������� �� ���������� � ���� �������� � ��� ����������
					for (int y = i; y < i + scale; y++)
						for (int x = j; x < j + scale; x++) {
							double value = X(d, y, x); // �������� �������� �������� �������
							mask(d, y, x) = 0; // �������� �����
							// ���� ������� �������� ������ �������������
							if (value > max) {
								max = value; // ��������� ��������
								imax = y; // ��������� ������ ������ ���������
								jmax = x; // ��������� ������ ������� ���������
							}
						}
					output(d, i / scale, j / scale) = max; // ���������� � �������� ������ ��������� ��������
					mask(d, imax, jmax) = 1; // ���������� 1 � ����� � ����� ������������ ������������� ��������
				}
		return output; // ���������� �������� ������
	}
	// �������� ���������������
	Tensor backward(const Tensor& dout, const Tensor& X) {
		Tensor dX(inputSize); // ������ ������ ��� ����������
		for (int d = 0; d < inputSize.depth; d++)
			for (int i = 0; i < inputSize.height; i++)
				for (int j = 0; j < inputSize.width; j++)
					dX(d, i, j) = dout(d, i / scale, j / scale) * mask(d, i, j); // �������� ��������� �� �����
		return dX; // ���������� ����������� ���������
	}
	// ������ ��������� �������
	TensorSize getOutputSize() const {
		return outputSize;
	}
	// ����� ���� � Json
	nlohmann::json getJson() {
		nlohmann::json js;

		// ������� ������
		js["inputSize"]["w"] = inputSize.width;
		js["inputSize"]["h"] = inputSize.height;
		js["inputSize"]["d"] = inputSize.depth;

		// �������� ������
		js["outputSize"]["w"] = outputSize.width;
		js["outputSize"]["h"] = outputSize.height;
		js["outputSize"]["d"] = outputSize.depth;

		js["scale"] = scale; // ����������� ����������

		js["mask"] = mask.getJson(); // �����

		return js;
	}
};



// ���� ��������� "�����������"
class cnn::ReLULayer
{
private:
	// ������ ����
	TensorSize size;

public:
	/*
	�����������-��������.
	������������ ������ ��� ������������� ���������� ������� ���������
	�������   ����.  ��  ��������   �������  ����������   �����������.
	*/
	ReLULayer() : ReLULayer(TensorSize(1, 1, 1)) {}
	// ����������� ����-���������� ReLU
	ReLULayer(TensorSize size) {
		this->size = size; // ��������� ������
	}
	// ����������� ����-���������� ReLU �� Json
	ReLULayer(nlohmann::json js) {
		size.width = js["size"]["w"].get<int>();
		size.height = js["size"]["h"].get<int>();
		size.depth = js["size"]["d"].get<int>();
	}
	// ������ ���������������
	Tensor forward(const Tensor& X) {
		Tensor output(size); // ������ �������� ������
		// ���������� �� ���� ��������� �������� �������
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					output(k, i, j) = X(k, i, j) > 0 ? X(k, i, j) : 0; // ��������� �������� ������� ���������
		return output; // ���������� �������� ������
	}
	// �������� ���������������
	Tensor backward(const Tensor& dout, const Tensor& X) {
		Tensor dX(size); // ������ ������ ����������
		// ���������� �� ���� ��������� ������� ����������
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					dX(k, i, j) = dout(k, i, j) * (X(k, i, j) > 0 ? 1 : 0); // �������� ��������� ���������� ���� �� ����������� ������� ���������
		return dX; // ���������� ������ ����������
	}
	// ������ ��������� �������
	TensorSize getOutputSize() const {
		return size;
	}
	// ����� ���� � Json
	nlohmann::json getJson() {
		nlohmann::json js;

		js["size"]["w"] = size.width;
		js["size"]["h"] = size.height;
		js["size"]["d"] = size.depth;

		return js;
	}
};



// ���� ��������� "��������"
class cnn::SigmoidLayer
{
private:
	// ������ ����
	TensorSize size;

public:
	/*
	�����������-��������.
	������������ ������ ��� ������������� ���������� ������� ���������
	�������   ����.  ��  ��������   �������  ����������   �����������.
	*/
	SigmoidLayer() : SigmoidLayer(TensorSize(1, 1, 1)) {}
	// ����������� ����-���������� sigmoid
	SigmoidLayer(TensorSize size) {
		this->size = size; // ��������� ������
	}
	// ����������� ����-���������� sigmoid �� Json
	SigmoidLayer(nlohmann::json js) {
		size.width = js["size"]["w"].get<int>();
		size.height = js["size"]["h"].get<int>();
		size.depth = js["size"]["d"].get<int>();
	}
	// ������ ���������������
	Tensor forward(const Tensor& X) {
		Tensor output(size); // ������ �������� ������
		// ���������� �� ���� ��������� �������� �������
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					output(k, i, j) = 1.0 / (1.0 + exp(-X(k, i, j))); // ��������� �������� ������� ���������
		return output; // ���������� �������� ������
	}
	// �������� ���������������
	Tensor backward(const Tensor& dout, const Tensor& X) {
		Tensor dX(size); // ������ ������ ����������
		// ���������� �� ���� ��������� ������� ����������
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					dX(k, i, j) = dout(k, i, j) * exp(-X(k, i, j)) / ((1.0 + exp(-X(k, i, j))) * (1.0 + exp(-X(k, i, j)))); // �������� ��������� ���������� ���� �� ����������� ������� ���������
		return dX; // ���������� ������ ����������
	}
	// ������ ��������� �������
	TensorSize getOutputSize() const {
		return size;
	}
	// ����� ���� � Json
	nlohmann::json getJson() {
		nlohmann::json js;

		js["size"]["w"] = size.width;
		js["size"]["h"] = size.height;
		js["size"]["d"] = size.depth;

		return js;
	}
};



// �������
class cnn::Matrix
{
private:
	// ����� �����
	int rows;
	// ����� ��������
	int columns;
	// ��������
	std::vector<double> values;

public:
	// �����������-��������.
	Matrix() : Matrix(1, 1) {}
	// ����������� �� �������� ��������
	Matrix(int rows, int columns) {
		this->rows = rows; // ��������� ����� �����
		this->columns = columns; // ��������� ����� ��������
		values = std::vector<double>((size_t)rows * columns, 0); // ������ ������� ��� �������� �������
	}
	// ����������� �� Json
	Matrix(nlohmann::json js) {
		rows = js["rows"].get<int>();
		columns = js["columns"].get<int>();
		values = std::vector<double>((size_t)rows * columns, 0);
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				values[(size_t)i * columns + j] = js["values"][(size_t)i * columns + j].get<double>();
	}
	// ����������
	double& operator()(int i, int j) {
		return values[(size_t)i * columns + j];
	}
	// ����������
	double operator()(int i, int j) const {
		return values[(size_t)i * columns + j];
	}
	// ��������� ��������
	void setValue(int i, int j, double value) {
		values[(size_t)i * columns + j] = value;
	}
	int size() {
		return (int)values.size();
	}
	// ����������
	double& operator[](int i) {
		return values[i];
	}
	// ����������
	double operator[](int i) const {
		return values[i];
	}
	// ����� � Json
	nlohmann::json getJson() {
		nlohmann::json js;

		js["rows"] = rows;
		js["columns"] = columns;
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				js["values"][(size_t)i * columns + j] = values[(size_t)i * columns + j];

		return js;
	}
};



// ������������ ����
class cnn::FullyConnectedLayer
{
private:
	// ��� ������������� �������
	enum class ActivationType
	{
		None, // ��� ���������
		Sigmoid, // ������������� �������
		Tanh, // ��������������� �������
		ReLU, // �����������
		LeakyReLU, // ����������� � �������
		ELU // ���������������� �����������
	};
	// ������ ������
	TensorSize inputSize;
	// �������� ������
	TensorSize outputSize;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution;

	// ����� ������� ��������
	int inputs;
	// ����� �������� ��������
	int outputs;
	// ��� ������������� �������
	ActivationType activationType;
	// ������ ����������� ������� ���������
	Tensor df;
	// ������� ������� �������������
	Matrix W;
	// ������� ���������� ������� �������������
	Matrix dW;
	// ��������
	std::vector<double> b;
	// ��������� ��������
	std::vector<double> db;

	// ��������� ���� ������������� ������� �� ������
	ActivationType getActivationType(const std::string& activationType) const {
		if (activationType == "sigmoid")
			return ActivationType::Sigmoid;
		if (activationType == "tanh")
			return ActivationType::Tanh;
		if (activationType == "relu")
			return ActivationType::ReLU;
		if (activationType == "leakyrelu")
			return ActivationType::LeakyReLU;
		if (activationType == "elu")
			return ActivationType::ELU;
		if (activationType == "none" || activationType.length() == 0)
			return ActivationType::None;
		throw std::runtime_error("Invalid activation function");
	}
	// ������������� ������� �������������
	void initWeights() {
		for (int i = 0; i < outputs; i++) {
			for (int j = 0; j < inputs; j++) {
				W(i, j) = distribution(generator);
			}
			b[i] = 0.01; // ��� �������� ������ ������� 0.01
		}
	}
	// ���������� ������������� �������
	void activate(Tensor& output) {
		switch (activationType) {
			case ActivationType::None:
				for (int i = 0; i < outputs; i++) {
					df[i] = 1;
				}
				break;
			case ActivationType::Sigmoid:
				for (int i = 0; i < outputs; i++) {
					output[i] = 1 / (1 + exp(-output[i]));
					df[i] = output[i] * (1 - output[i]);
				}
				break;
			case ActivationType::Tanh:
				for (int i = 0; i < outputs; i++) {
					output[i] = tanh(output[i]);
					df[i] = 1 - output[i] * output[i];
				}
				break;
			case  ActivationType::ReLU:
				for (int i = 0; i < outputs; i++) {
					if (output[i] <= 0) {
						output[i] = 0;
						df[i] = 0;
					}
					else
						df[i] = 1;
				}
				break;
			case ActivationType::LeakyReLU:
				for (int i = 0; i < outputs; i++) {
					if (output[i] <= 0) {
						output[i] *= 0.01;
						df[i] = 0.01;
					}
					else
						df[i] = 1;
				}
				break;
			case ActivationType::ELU:
				for (int i = 0; i < outputs; i++) {
					if (output[i] <= 0) {
						output[i] = exp(output[i]) - 1;
						df[i] = output[i] + 1;
					}
					else
						df[i] = 1;
				}
				break;
		}

	}

public:
	/*
	�����������-��������.
	������������ ������ ��� ������������� ���������� ������� ���������
	�������   ����.  ��  ��������   �������  ����������   �����������.
	*/
	FullyConnectedLayer() : FullyConnectedLayer(TensorSize(1, 1, 1), 1) {}
	// ����������� ������������� ����
	FullyConnectedLayer(TensorSize size, int outputs, const std::string& activationType = "none") : // �������� ����
		distribution(0.0, sqrt(2.0 / ((size_t)size.height * size.width * size.depth))),
		df(1, 1, outputs),
		W(outputs, size.height* size.width* size.depth),
		dW(outputs, size.height* size.width* size.depth),
		b(outputs),
		db(outputs)
	{
		// ���������� ������� ������
		inputSize.width = size.width;
		inputSize.height = size.height;
		inputSize.depth = size.depth;

		// ��������� �������� ������
		outputSize.width = 1;
		outputSize.height = 1;
		outputSize.depth = outputs;

		inputs = size.height * size.width * size.depth; // ���������� ����� ������� ��������
		this->outputs = outputs; // ���������� ����� �������� ��������

		this->activationType = getActivationType(activationType); // �������� ������������� �������

		initWeights(); // �������������� ������� ������������		
	}
	// ����������� ������������� ���� �� Json
	FullyConnectedLayer(nlohmann::json js) {
		// ������� ������
		inputSize.width = js["inputSize"]["w"].get<int>();
		inputSize.height = js["inputSize"]["h"].get<int>();
		inputSize.depth = js["inputSize"]["d"].get<int>();

		// �������� ������
		outputSize.width = js["outputSize"]["w"].get<int>();
		outputSize.height = js["outputSize"]["h"].get<int>();
		outputSize.depth = js["outputSize"]["d"].get<int>();

		inputs = js["inputs"].get<int>();
		outputs = js["outputs"].get<int>(); // ����� �������� ��������
		
		df = Tensor(js["df"]);
		W = Matrix(js["W"]);
		dW = Matrix(outputs, inputSize.height * inputSize.width * inputSize.depth);

		activationType = js["activationType"].get<ActivationType>(); // ������������� �������

		b = js["b"].get<std::vector<double>>(); // ������ ��������
		db = std::vector<double>(outputs); // ������ ������ ���������� �� ����� ��������
	}
	// ������ ���������������
	// ������ �������� ��� ���������� ������
	Tensor forward(const Tensor& X) {
		Tensor output(outputSize); // ������ �������� ������
		// ���������� �� ������� ��������� �������
		for (int i = 0; i < outputs; i++) {
			double sum = b[i]; // ���������� ��������
			// �������� ������� ������ �� �������
			for (int j = 0; j < inputs; j++)
				sum += W(i, j) * X[j];
			output[i] = sum;
		}
		activate(output); // ��������� ������������� �������
		return output; // ���������� �������� ������
	}
	// �������� ���������������
	Tensor backward(const Tensor& dout, const Tensor& X) {
		// ��������� ����������� �� ��������� ���������� ���� ��� ���������� ���������� ���������
		for (int i = 0; i < outputs; i++)
			df[i] *= dout[i];
		// ��������� ��������� �� ������� �������������
		for (int i = 0; i < outputs; i++) {
			for (int j = 0; j < inputs; j++)
				dW(i, j) = df[i] * X[j];
			db[i] = df[i];
		}
		Tensor dX(inputSize); // ������ ������ ��� ���������� �� ������
		// ��������� ��������� �� ������
		for (int j = 0; j < inputs; j++) {
			double sum = 0;
			for (int i = 0; i < outputs; i++)
				sum += W(i, j) * df[i];
			dX[j] = sum; // ���������� ��������� � ������ ����������
		}
		return dX; // ���������� ������ ����������
	}
	// ���������� ������� �������������
	void updateWeights(double learning_rate) {
		for (int i = 0; i < outputs; i++) {
			for (int j = 0; j < inputs; j++)
				W(i, j) -= learning_rate * dW(i, j);
			b[i] -= learning_rate * db[i]; // ��������� ���� ��������
		}
	}
	// ��������� ���� �������
	void setWeight(int i, int j, double weight) {
		W(i, j) = weight;
	}
	// ��������� ���� ��������
	void setBias(int i, double bias) {
		b[i] = bias;
	}
	// ������ ��������� �������
	TensorSize getOutputSize() const {
		return outputSize;
	}
	Matrix getWeights() {
		return W;
	}
	std::vector<double> getBias() {
		return b;
	}
	// ����� � Json
	nlohmann::json getJson() {
		nlohmann::json js;

		js["distribution"]["mean"] = distribution.mean();
		js["distribution"]["sigma"] = distribution.sigma();

		js["df"] = df.getJson();
		js["W"] = W.getJson();

		// ������� ������
		js["inputSize"]["w"] = inputSize.width;
		js["inputSize"]["h"] = inputSize.height;
		js["inputSize"]["d"] = inputSize.depth;

		// �������� ������
		js["outputSize"]["w"] = outputSize.width;
		js["outputSize"]["h"] = outputSize.height;
		js["outputSize"]["d"] = outputSize.depth;

		js["inputs"] = inputs;
		js["outputs"] = outputs; // ����� �������� ��������

		js["activationType"] = activationType; // ������������� �������

		js["b"] = b; // ������ ��������

		return js;
	}
};