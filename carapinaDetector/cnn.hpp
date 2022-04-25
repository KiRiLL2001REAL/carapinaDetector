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



// размерность тензора
struct cnn::TensorSize
{
	int depth; // глубина
	int height; // высота
	int width; // ширина

	TensorSize() {
		width = height = depth = 1;
	}
	TensorSize(int width, int height, int depth) {
		this->width = width;
		this->height = height;
		this->depth = depth;
	}
};



// тензор
class cnn::Tensor
{
private:
	// размерность тензора
	TensorSize size;
	// значения тензора
	std::vector<double> values;
	// произведение глубины на ширину для индексации
	int dw;

	// инициализация по размерам
	void init(int width, int height, int depth) {
		size.width = width; // запоминаем ширину
		size.height = height; // запоминаем высоту
		size.depth = depth; // запоминаем глубину

		dw = depth * width; // запоминаем произведение глубины на ширину для индексации

		values = std::vector<double>((size_t)width * height * depth, 0); // создаём вектор из width * height * depth нулей
	}

public:
	// Конструктор-заглушка.
	Tensor() : Tensor(1, 1, 1) {}
	// создание из размеров
	Tensor(int width, int height, int depth) {
		init(width, height, depth);
	}
	// создание из размера
	Tensor(const TensorSize& size) {
		init(size.width, size.height, size.depth);
	}
	// создание из json
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

	// индексация
	double& operator()(int d, int i, int j) {
		return values[(size_t)i * dw + (size_t)j * size.depth + d];
	}
	// индексация
	double operator()(int d, int i, int j) const {
		return values[(size_t)i * dw + (size_t)j * size.depth + d];
	}
	// индексация
	double& operator[](int i) {
		return values[i];
	}
	// индексация
	double operator[](int i) const {
		return values[i];
	}
	// получение размера
	TensorSize getSize() const {
		return size;
	}
	// вывод тензора
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
	// установка значения
	void setValue(int d, int i, int j, double value) {
		values[(size_t)i * dw + (size_t)j * size.depth + d] = value;
	}
	// вывод тензора в json
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



// слой свёртки
class cnn::ConvLayer
{
private:
	// генератор случайных чисел
	std::default_random_engine generator;
	// с нормальным распределением
	std::normal_distribution<double> distribution;
	// размер входа
	TensorSize inputSize;
	// размер выхода
	TensorSize outputSize;
	// фильтры
	std::vector<Tensor> W;
	// смещения
	std::vector<double> b;
	// градиенты фильтров
	std::vector<Tensor> dW;
	// градиенты смещений
	std::vector<double> db;
	// дополнение нулями
	int P;
	// шаг свёртки
	int S;
	// количество фильтров
	int fc;
	// размер фильтров
	int fs;
	// глубина фильтров
	int fd;

	// инициализация весовых коэффициентов
	void initWeights() {
		// проходимся по каждому из фильтров
		for (int index = 0; index < fc; index++) {
			for (int i = 0; i < fs; i++) {
				for (int j = 0; j < fs; j++) {
					for (int k = 0; k < fd; k++) {
						W[index](k, i, j) = distribution(generator); // генерируем случайное число и записываем его в элемент фильтра
					}
				}
			}
			b[index] = 0.01; // все смещения устанавливаем в 0.01
		}
	};

public:
	/*
	Конструктор-заглушка.
	Использовать только при необходимости объявления массива элементов
	данного   типа.  Не  забудьте   вызвать  нормальный   конструктор.
	*/
	ConvLayer() : ConvLayer(TensorSize(1, 1, 1), 1, 1, 1, 1) {}
	// конструктор свёрточного слоя
	ConvLayer(TensorSize size, int fc, int fs, int P, int S) :
		distribution(0.0, sqrt(2.0 / ((size_t)fs * fs * size.depth)))
	{
		// запоминаем входной размер
		inputSize.width = size.width;
		inputSize.height = size.height;
		inputSize.depth = size.depth;

		// вычисляем выходной размер
		outputSize.width = (size.width - fs + 2 * P) / S + 1;
		outputSize.height = (size.height - fs + 2 * P) / S + 1;
		outputSize.depth = fc;

		this->P = P; // сохраняем дополнение нулями
		this->S = S; // сохраняем шаг свёртки

		this->fc = fc; // сохраняем число фильтров
		this->fs = fs; // сохраняем размер фильтров
		this->fd = size.depth; // сохраняем глубину фильтров

		// добавляем fc тензоров для весов фильтров и их градиентов
		W = std::vector<Tensor>(fc, Tensor(fs, fs, fd));
		dW = std::vector<Tensor>(fc, Tensor(fs, fs, fd));

		// добавляем fc нулей для весов смещения и их градиентов
		b = std::vector<double>(fc, 0);
		db = std::vector<double>(fc, 0);

		initWeights(); // инициализируем весовые коэффициенты
	}
	// конструктор свёрточного слоя из Json
	ConvLayer(nlohmann::json js) {
		// входной размер
		inputSize.width = js["inputSize"]["w"].get<int>();
		inputSize.height = js["inputSize"]["h"].get<int>();
		inputSize.depth = js["inputSize"]["d"].get<int>();

		// выходной размер
		outputSize.width = js["outputSize"]["w"].get<int>();
		outputSize.height = js["outputSize"]["h"].get<int>();
		outputSize.depth = js["outputSize"]["d"].get<int>();

		P = js["P"].get<int>(); // дополнение нулями
		S = js["S"].get<int>(); // шаг свёртки

		fc = js["fc"].get<int>(); // число фильтров
		fs = js["fs"].get<int>(); // размер фильтров
		fd = js["fd"].get<int>(); // глубина фильтров

		// fc тензоров весов фильтров и их градиентов
		W = std::vector<Tensor>(fc);
		for (int i = 0; i < fc; i++)
			W[i] = Tensor(js["W"][i]);
		dW = std::vector<Tensor>(fc, Tensor(fs, fs, fd));

		// fc нулей весов смещения и их градиентов
		b = std::vector<double>(fc);
		for (int i = 0; i < fc; i++)
			b[i] = js["b"][i].get<double>();
		db = std::vector<double>(fc, 0);
	}
	// прямое распространение сигнала
	Tensor forward(const Tensor& X) {
		Tensor output(outputSize); // создаём выходной тензор
		// проходимся по каждому из фильтров
		for (int f = 0; f < fc; f++) {
			for (int y = 0; y < outputSize.height; y++) {
				for (int x = 0; x < outputSize.width; x++) {
					double sum = b[f]; // сразу прибавляем смещение
					// проходимся по фильтрам
					for (int i = 0; i < fs; i++) {
						for (int j = 0; j < fs; j++) {
							int i0 = S * y + i - P;
							int j0 = S * x + j - P;
							// поскольку вне границ входного тензора элементы нулевые, то просто игнорируем их
							if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
								continue;
							// проходимся по всей глубине тензора и считаем сумму
							for (int c = 0; c < fd; c++)
								sum += X(c, i0, j0) * W[f](c, i, j);
						}
					}
					output(f, y, x) = sum; // записываем результат свёртки в выходной тензор
				}
			}
		}

		return output; // возвращаем выходной тензор
	}
	// обратное распространение ошибки
	Tensor backward(const Tensor& dout, const Tensor& X) {
		TensorSize size; // размер дельт
		// расчитываем размер для дельт
		size.height = S * (outputSize.height - 1) + 1;
		size.width = S * (outputSize.width - 1) + 1;
		size.depth = outputSize.depth;
		Tensor deltas(size); // создаём тензор для дельт
		// расчитываем значения дельт
		for (int d = 0; d < size.depth; d++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					deltas(d, i * S, j * S) = dout(d, i, j);
		// расчитываем градиенты весов фильтров и смещений
		for (int f = 0; f < fc; f++)
			for (int y = 0; y < size.height; y++)
				for (int x = 0; x < size.width; x++) {
					double delta = deltas(f, y, x); // запоминаем значение градиента
					for (int i = 0; i < fs; i++) {
						for (int j = 0; j < fs; j++) {
							int i0 = i + y - P;
							int j0 = j + x - P;
							// игнорируем выходящие за границы элементы
							if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
								continue;
							// наращиваем градиент фильтра
							for (int c = 0; c < fd; c++)
								dW[f](c, i, j) += X(c, i0, j0) * delta;
						}
					}
					db[f] += delta; // наращиваем градиент смещения
				}
		int pad = fs - 1 - P; // заменяем величину дополнения
		Tensor dX(inputSize); // создаём тензор градиентов по входу
		// расчитываем значения градиента
		for (int y = 0; y < inputSize.height; y++)
			for (int x = 0; x < inputSize.width; x++)
				for (int c = 0; c < fd; c++) {
					double sum = 0; // сумма для градиента
					// идём по всем весовым коэффициентам фильтров
					for (int i = 0; i < fs; i++) {
						for (int j = 0; j < fs; j++) {
							int i0 = y + i - pad;
							int j0 = x + j - pad;
							// игнорируем выходящие за границы элементы
							if (i0 < 0 || i0 >= size.height || j0 < 0 || j0 >= size.width)
								continue;
							// суммируем по всем фильтрам
							for (int f = 0; f < fc; f++)
								sum += W[f](c, fs - 1 - i, fs - 1 - j) * deltas(f, i0, j0); // добавляем произведение повёрнутых фильтров на дельты
						}
					}
					dX(c, y, x) = sum; // записываем результат в тензор градиента
				}
		return dX; // возвращаем тензор градиентов
	}
	// обновление весовых коэффициентов
	void updateWeights(double learning_rate) {
		for (int index = 0; index < fc; index++) {
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					for (int d = 0; d < fd; d++) {
						W[index](d, i, j) -= learning_rate * dW[index](d, i, j); // вычитаем градиент, умноженный на скорость обучения
						dW[index](d, i, j) = 0; // обнуляем градиент фильтра
					}
			b[index] -= learning_rate * db[index]; // вычитаем градиент, умноженный на скорость обучения
			db[index] = 0; // обнуляем градиент веса смещения
		}
	}
	// установка веса фильтра по индексу
	void setWeight(int index, int d, int i, int j, double weight) {
		W[index](d, i, j) = weight;
	}
	// установка веса смещения по индексу
	void setBias(int index, double bias) {
		b[index] = bias;
	}
	// размер выходного тензора
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
	// вывод слоя в json
	nlohmann::json getJson() {
		nlohmann::json js;

		// входной размер
		js["inputSize"]["w"] = inputSize.width;
		js["inputSize"]["h"] = inputSize.height;
		js["inputSize"]["d"] = inputSize.depth;

		// выходной размер
		js["outputSize"]["w"] = outputSize.width;
		js["outputSize"]["h"] = outputSize.height;
		js["outputSize"]["d"] = outputSize.depth;

		js["P"] = P; // дополнение нулями
		js["S"] = S; // шаг свёртки

		js["fc"] = fc; // число фильтров
		js["fs"] = fs; // размер фильтров
		js["fd"] = fd; // глубинf фильтров

		// fc тензоров весов фильтров и их градиентов
		for (int i = 0; i < fc; i++)
			js["W"][i] = W[i].getJson();

		// fc нулей весов смещения и их градиентов
		for (int i = 0; i < fc; i++)
			js["b"][i] = b[i];

		return js;
	}
};



// слой max pooling'а
class cnn::MaxPoolingLayer
{
private:
	// размер входа
	TensorSize inputSize;
	// размер выхода
	TensorSize outputSize;
	// во сколько раз уменьшается размерность
	int scale;
	// маска для максимумов
	Tensor mask;

public:
	/*
	Конструктор-заглушка.
	Использовать только при необходимости объявления массива элементов
	данного   типа.  Не  забудьте   вызвать  нормальный   конструктор.
	*/
	MaxPoolingLayer() : MaxPoolingLayer(TensorSize(1, 1, 1)) {}
	// конструктор maxpool слоя
	MaxPoolingLayer(TensorSize size, int scale = 2) :
		mask(size)
	{
		// запоминаем входной размер
		inputSize.width = size.width;
		inputSize.height = size.height;
		inputSize.depth = size.depth;

		// вычисляем выходной размер
		outputSize.width = size.width / scale;
		outputSize.height = size.height / scale;
		outputSize.depth = size.depth;

		this->scale = scale; // запоминаем коэффициент уменьшения

		// заполняем маски единицами
		TensorSize mask_size = mask.getSize();
		for (int d = 0; d < mask_size.depth; d++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					mask(d, i, j) = (double)rand() / RAND_MAX;
	}
	// конструктор maxpool слоя из Json
	MaxPoolingLayer(nlohmann::json js) {
		// входной размер
		inputSize.width = js["inputSize"]["w"].get<int>();
		inputSize.height = js["inputSize"]["h"].get<int>();
		inputSize.depth = js["inputSize"]["d"].get<int>();

		// выходной размер
		outputSize.width = js["outputSize"]["w"].get<int>();
		outputSize.height = js["outputSize"]["h"].get<int>();
		outputSize.depth = js["outputSize"]["d"].get<int>();

		scale = js["scale"].get<int>(); // коэффициент уменьшения

		mask = Tensor(js["mask"]); // маска
	}
	// прямое распространение
	Tensor forward(const Tensor& X) {
		Tensor output(outputSize); // создаём выходной тензор
		// проходимся по каждому из каналов
		for (int d = 0; d < inputSize.depth; d++)
			for (int i = 0; i < inputSize.height; i += scale)
				for (int j = 0; j < inputSize.width; j += scale) {
					int imax = i; // индекс строки максимума
					int jmax = j; // индекс столбца максимума
					double max = X(d, i, j); // начальное значение максимума - значение первой клетки подматрицы
					// проходимся по подматрице и ищем максимум и его координаты
					for (int y = i; y < i + scale; y++)
						for (int x = j; x < j + scale; x++) {
							double value = X(d, y, x); // получаем значение входного тензора
							mask(d, y, x) = 0; // обнуляем маску
							// если входное значение больше максимального
							if (value > max) {
								max = value; // обновляем максимум
								imax = y; // обновляем индекс строки максимума
								jmax = x; // обновляем индекс столбца максимума
							}
						}
					output(d, i / scale, j / scale) = max; // записываем в выходной тензор найденный максимум
					mask(d, imax, jmax) = 1; // записываем 1 в маску в месте расположения максимального элемента
				}
		return output; // возвращаем выходной тензор
	}
	// обратное распространение
	Tensor backward(const Tensor& dout, const Tensor& X) {
		Tensor dX(inputSize); // создаём тензор для градиентов
		for (int d = 0; d < inputSize.depth; d++)
			for (int i = 0; i < inputSize.height; i++)
				for (int j = 0; j < inputSize.width; j++)
					dX(d, i, j) = dout(d, i / scale, j / scale) * mask(d, i, j); // умножаем градиенты на маску
		return dX; // возвращаем посчитанные градиенты
	}
	// размер выходного тензора
	TensorSize getOutputSize() const {
		return outputSize;
	}
	// вывод слоя в Json
	nlohmann::json getJson() {
		nlohmann::json js;

		// входной размер
		js["inputSize"]["w"] = inputSize.width;
		js["inputSize"]["h"] = inputSize.height;
		js["inputSize"]["d"] = inputSize.depth;

		// выходной размер
		js["outputSize"]["w"] = outputSize.width;
		js["outputSize"]["h"] = outputSize.height;
		js["outputSize"]["d"] = outputSize.depth;

		js["scale"] = scale; // коэффициент уменьшения

		js["mask"] = mask.getJson(); // маска

		return js;
	}
};



// слой активации "выпрямитель"
class cnn::ReLULayer
{
private:
	// размер слоя
	TensorSize size;

public:
	/*
	Конструктор-заглушка.
	Использовать только при необходимости объявления массива элементов
	данного   типа.  Не  забудьте   вызвать  нормальный   конструктор.
	*/
	ReLULayer() : ReLULayer(TensorSize(1, 1, 1)) {}
	// конструктор слоя-активатора ReLU
	ReLULayer(TensorSize size) {
		this->size = size; // сохраняем размер
	}
	// конструктор слоя-активатора ReLU из Json
	ReLULayer(nlohmann::json js) {
		size.width = js["size"]["w"].get<int>();
		size.height = js["size"]["h"].get<int>();
		size.depth = js["size"]["d"].get<int>();
	}
	// прямое распространение
	Tensor forward(const Tensor& X) {
		Tensor output(size); // создаём выходной тензор
		// проходимся по всем значениям входного тензора
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					output(k, i, j) = X(k, i, j) > 0 ? X(k, i, j) : 0; // вычисляем значение функции активации
		return output; // возвращаем выходной тензор
	}
	// обратное распространение
	Tensor backward(const Tensor& dout, const Tensor& X) {
		Tensor dX(size); // создаём тензор градиентов
		// проходимся по всем значениям тензора градиентов
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					dX(k, i, j) = dout(k, i, j) * (X(k, i, j) > 0 ? 1 : 0); // умножаем градиенты следующего слоя на производную функции активации
		return dX; // возвращаем тензор градиентов
	}
	// размер выходного тензора
	TensorSize getOutputSize() const {
		return size;
	}
	// вывод слоя в Json
	nlohmann::json getJson() {
		nlohmann::json js;

		js["size"]["w"] = size.width;
		js["size"]["h"] = size.height;
		js["size"]["d"] = size.depth;

		return js;
	}
};



// слой активации "сигмоида"
class cnn::SigmoidLayer
{
private:
	// размер слоя
	TensorSize size;

public:
	/*
	Конструктор-заглушка.
	Использовать только при необходимости объявления массива элементов
	данного   типа.  Не  забудьте   вызвать  нормальный   конструктор.
	*/
	SigmoidLayer() : SigmoidLayer(TensorSize(1, 1, 1)) {}
	// конструктор слоя-активатора sigmoid
	SigmoidLayer(TensorSize size) {
		this->size = size; // сохраняем размер
	}
	// конструктор слоя-активатора sigmoid из Json
	SigmoidLayer(nlohmann::json js) {
		size.width = js["size"]["w"].get<int>();
		size.height = js["size"]["h"].get<int>();
		size.depth = js["size"]["d"].get<int>();
	}
	// прямое распространение
	Tensor forward(const Tensor& X) {
		Tensor output(size); // создаём выходной тензор
		// проходимся по всем значениям входного тензора
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					output(k, i, j) = 1.0 / (1.0 + exp(-X(k, i, j))); // вычисляем значение функции активации
		return output; // возвращаем выходной тензор
	}
	// обратное распространение
	Tensor backward(const Tensor& dout, const Tensor& X) {
		Tensor dX(size); // создаём тензор градиентов
		// проходимся по всем значениям тензора градиентов
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				for (int k = 0; k < size.depth; k++)
					dX(k, i, j) = dout(k, i, j) * exp(-X(k, i, j)) / ((1.0 + exp(-X(k, i, j))) * (1.0 + exp(-X(k, i, j)))); // умножаем градиенты следующего слоя на производную функции активации
		return dX; // возвращаем тензор градиентов
	}
	// размер выходного тензора
	TensorSize getOutputSize() const {
		return size;
	}
	// вывод слоя в Json
	nlohmann::json getJson() {
		nlohmann::json js;

		js["size"]["w"] = size.width;
		js["size"]["h"] = size.height;
		js["size"]["d"] = size.depth;

		return js;
	}
};



// матрица
class cnn::Matrix
{
private:
	// число строк
	int rows;
	// число столбцов
	int columns;
	// значения
	std::vector<double> values;

public:
	// Конструктор-заглушка.
	Matrix() : Matrix(1, 1) {}
	// конструктор из заданных размеров
	Matrix(int rows, int columns) {
		this->rows = rows; // сохраняем число строк
		this->columns = columns; // сохраняем число столбцов
		values = std::vector<double>((size_t)rows * columns, 0); // создаём векторы для значений матрицы
	}
	// конструктор из Json
	Matrix(nlohmann::json js) {
		rows = js["rows"].get<int>();
		columns = js["columns"].get<int>();
		values = std::vector<double>((size_t)rows * columns, 0);
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				values[(size_t)i * columns + j] = js["values"][(size_t)i * columns + j].get<double>();
	}
	// индексация
	double& operator()(int i, int j) {
		return values[(size_t)i * columns + j];
	}
	// индексация
	double operator()(int i, int j) const {
		return values[(size_t)i * columns + j];
	}
	// установка значения
	void setValue(int i, int j, double value) {
		values[(size_t)i * columns + j] = value;
	}
	int size() {
		return (int)values.size();
	}
	// индексация
	double& operator[](int i) {
		return values[i];
	}
	// индексация
	double operator[](int i) const {
		return values[i];
	}
	// вывод в Json
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



// полносвязный слой
class cnn::FullyConnectedLayer
{
private:
	// тип активационной функции
	enum class ActivationType
	{
		None, // без активации
		Sigmoid, // сигмоидальная функция
		Tanh, // гиперболический тангенс
		ReLU, // выпрямитель
		LeakyReLU, // выпрямитель с утечкой
		ELU // экспоненциальный выпрямитель
	};
	// входой размер
	TensorSize inputSize;
	// выходной размер
	TensorSize outputSize;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution;

	// число входных нейронов
	int inputs;
	// число выходных нейронов
	int outputs;
	// тип активационной функции
	ActivationType activationType;
	// тензор производных функции активации
	Tensor df;
	// матрица весовых коэффициентов
	Matrix W;
	// матрица градиентов весовых коэффициентов
	Matrix dW;
	// смещения
	std::vector<double> b;
	// градиенты смещений
	std::vector<double> db;

	// получение типа активационной функции по строке
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
	// инициализация весовых коэффициентов
	void initWeights() {
		for (int i = 0; i < outputs; i++) {
			for (int j = 0; j < inputs; j++) {
				W(i, j) = distribution(generator);
			}
			b[i] = 0.01; // все смещения делаем равными 0.01
		}
	}
	// применение активационной функции
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
	Конструктор-заглушка.
	Использовать только при необходимости объявления массива элементов
	данного   типа.  Не  забудьте   вызвать  нормальный   конструктор.
	*/
	FullyConnectedLayer() : FullyConnectedLayer(TensorSize(1, 1, 1), 1) {}
	// конструктор полносвязного слоя
	FullyConnectedLayer(TensorSize size, int outputs, const std::string& activationType = "none") : // создание слоя
		distribution(0.0, sqrt(2.0 / ((size_t)size.height * size.width * size.depth))),
		df(1, 1, outputs),
		W(outputs, size.height* size.width* size.depth),
		dW(outputs, size.height* size.width* size.depth),
		b(outputs),
		db(outputs)
	{
		// запоминаем входной размер
		inputSize.width = size.width;
		inputSize.height = size.height;
		inputSize.depth = size.depth;

		// вычисляем выходной размер
		outputSize.width = 1;
		outputSize.height = 1;
		outputSize.depth = outputs;

		inputs = size.height * size.width * size.depth; // запоминаем число входных нейронов
		this->outputs = outputs; // запоминаем число выходных нейронов

		this->activationType = getActivationType(activationType); // получаем активационную функцию

		initWeights(); // инициализируем весовые коэффициенты		
	}
	// конструктор полносвязного слоя из Json
	FullyConnectedLayer(nlohmann::json js) {
		// входной размер
		inputSize.width = js["inputSize"]["w"].get<int>();
		inputSize.height = js["inputSize"]["h"].get<int>();
		inputSize.depth = js["inputSize"]["d"].get<int>();

		// выходной размер
		outputSize.width = js["outputSize"]["w"].get<int>();
		outputSize.height = js["outputSize"]["h"].get<int>();
		outputSize.depth = js["outputSize"]["d"].get<int>();

		inputs = js["inputs"].get<int>();
		outputs = js["outputs"].get<int>(); // число выходных нейронов
		
		df = Tensor(js["df"]);
		W = Matrix(js["W"]);
		dW = Matrix(outputs, inputSize.height * inputSize.width * inputSize.depth);

		activationType = js["activationType"].get<ActivationType>(); // активационная функция

		b = js["b"].get<std::vector<double>>(); // вектор смещений
		db = std::vector<double>(outputs); // создаём вектор градиентов по весам смещения
	}
	// прямое распространение
	// тензор читается как одномерный массив
	Tensor forward(const Tensor& X) {
		Tensor output(outputSize); // создаём выходной тензор
		// проходимся по каждому выходному нейрону
		for (int i = 0; i < outputs; i++) {
			double sum = b[i]; // прибавляем смещение
			// умножаем входной тензор на матрицу
			for (int j = 0; j < inputs; j++)
				sum += W(i, j) * X[j];
			output[i] = sum;
		}
		activate(output); // применяем активационную функцию
		return output; // возвращаем выходной тензор
	}
	// обратное распространение
	Tensor backward(const Tensor& dout, const Tensor& X) {
		// домножаем производные на градиенты следующего слоя для сокращения количества умножений
		for (int i = 0; i < outputs; i++)
			df[i] *= dout[i];
		// вычисляем градиенты по весовым коэффициентам
		for (int i = 0; i < outputs; i++) {
			for (int j = 0; j < inputs; j++)
				dW(i, j) = df[i] * X[j];
			db[i] = df[i];
		}
		Tensor dX(inputSize); // создаём тензор для градиентов по входам
		// вычисляем градиенты по входам
		for (int j = 0; j < inputs; j++) {
			double sum = 0;
			for (int i = 0; i < outputs; i++)
				sum += W(i, j) * df[i];
			dX[j] = sum; // записываем результат в тензор градиентов
		}
		return dX; // возвращаем тензор градиентов
	}
	// обновление весовых коэффициентов
	void updateWeights(double learning_rate) {
		for (int i = 0; i < outputs; i++) {
			for (int j = 0; j < inputs; j++)
				W(i, j) -= learning_rate * dW(i, j);
			b[i] -= learning_rate * db[i]; // обновляем веса смещения
		}
	}
	// установка веса матрицы
	void setWeight(int i, int j, double weight) {
		W(i, j) = weight;
	}
	// установка веса смещения
	void setBias(int i, double bias) {
		b[i] = bias;
	}
	// размер выходного тензора
	TensorSize getOutputSize() const {
		return outputSize;
	}
	Matrix getWeights() {
		return W;
	}
	std::vector<double> getBias() {
		return b;
	}
	// вывод в Json
	nlohmann::json getJson() {
		nlohmann::json js;

		js["distribution"]["mean"] = distribution.mean();
		js["distribution"]["sigma"] = distribution.sigma();

		js["df"] = df.getJson();
		js["W"] = W.getJson();

		// входной размер
		js["inputSize"]["w"] = inputSize.width;
		js["inputSize"]["h"] = inputSize.height;
		js["inputSize"]["d"] = inputSize.depth;

		// выходной размер
		js["outputSize"]["w"] = outputSize.width;
		js["outputSize"]["h"] = outputSize.height;
		js["outputSize"]["d"] = outputSize.depth;

		js["inputs"] = inputs;
		js["outputs"] = outputs; // число выходных нейронов

		js["activationType"] = activationType; // активационная функция

		js["b"] = b; // вектор смещений

		return js;
	}
};