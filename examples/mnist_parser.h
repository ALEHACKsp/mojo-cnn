// == mojo ====================================================================
//
//    mnist_parser.h: prepares MNIST data for testing/training
//
//    This code was modified from tiny_cnn https://github.com/nyanp/tiny-cnn
//    It can parse MNIST data which you need to download and unzip locally on
//    your machine.
//    You can get it from: http://yann.lecun.com/exdb/mnist/index.html
//
// ==================================================================== mojo ==

/*
	Copyright (c) 2013, Taiga Nomi
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
	* Neither the name of the <organization> nor the
	names of its contributors may be used to endorse or promote products
	derived from this software without specific prior written permission.
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <iostream> // cout
#include <sstream>
#include <fstream>
#include <iomanip> //setw
#include <random>
#include <stdio.h>

namespace mnist
{
	// 返回数据集名称
	std::string data_name() { return std::string("MNIST"); }

	// 字节转换
	template<typename T = int>
	T* reverse_endian(T* p)
	{
		std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
		return p;
	}

	/// <summary>
	/// 解析MNIST数据集的标签.
	/// </summary>
	/// <param name="label_file">标签文件路径.</param>
	/// <param name="labels">返回解析出来的标签.</param>
	/// <returns></returns>
	bool parse_mnist_labels(const std::string& label_file, std::vector<int> *labels)
	{
		//以二进制方式读取文件
		std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
		if (ifs.is_open() == false) return false;

		//读取标志位和标签数量
		int magic_number = 0, num_items = 0;
		ifs.read((char*)&magic_number, 4);
		ifs.read((char*)&num_items, 4);

		//进行字节转换
		reverse_endian(&magic_number);
		reverse_endian(&num_items);

		//读取标签数据
		for (int i = 0; i < num_items; i++)
		{
			unsigned char label = 0;
			ifs.read((char*)&label, 1);
			labels->push_back((int)label);
		}

		//关闭标签文件
		ifs.close();

		//返回
		return true;
	}

	// MNIST数据集头结构
	struct mnist_header
	{
		int magic_number;	//标识符
		int num_items;			//图像数量
		int num_rows;			//图像宽度
		int num_cols;			//图像高度
	};

	/// <summary>
	/// 解析MNIST图像数据.
	/// </summary>
	/// <param name="image_file">MNIST图像文件路径.</param>
	/// <param name="images">返回读取到的图像数据.</param>
	/// <param name="scale_min">最小缩放.</param>
	/// <param name="scale_max">最大缩放.</param>
	/// <param name="x_padding">X轴填充.</param>
	/// <param name="y_padding">Y轴填充.</param>
	/// <returns></returns>
	bool parse_mnist_images(const std::string& image_file,
		std::vector<std::vector<float>> *images,
		float scale_min = -1.0, float scale_max = 1.0,
		int x_padding = 0, int y_padding = 0)
	{
		//以二进制方式打开图像文件
		std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
		if (ifs.is_open() == false) return false;

		mnist_header header{ 0 };

		// 读取文件头信息
		ifs.read((char*)&header.magic_number, 4);
		ifs.read((char*)&header.num_items, 4);
		ifs.read((char*)&header.num_rows, 4);
		ifs.read((char*)&header.num_cols, 4);

		// 字节转换
		reverse_endian(&header.magic_number);
		reverse_endian(&header.num_items);
		reverse_endian(&header.num_rows);
		reverse_endian(&header.num_cols);

		/*
		为什么都要乘以2呢?
		因为图像有左边和右边呀,上边和下边呀,不能只填充左边而不填充右边吧?
		*/
		const int width = header.num_cols + 2 * x_padding;
		const int height = header.num_rows + 2 * y_padding;

		// 读取每一张图像
		for (int i = 0; i < header.num_items; i++)
		{
			// 读取原始图像数据
			std::vector<unsigned char> image_vec(header.num_rows * header.num_cols);
			ifs.read((char*)&image_vec[0], header.num_rows * header.num_cols);

			// 存放归一化的图像数据
			std::vector<float> image;
			image.resize(width * height, scale_min);

			// 图像处理
			for (int y = 0; y < header.num_rows; y++)
			{
				for (int x = 0; x < header.num_cols; x++)
					image[width * (y + y_padding) + x + x_padding] =
					(image_vec[y * header.num_cols + x] / 255.0f) * //缩放到(0,1)之间
					(scale_max - scale_min) +
					scale_min;
			}

			// 保存图像
			images->push_back(std::move(image));
		}

		//关闭文件
		ifs.close();

		//返回
		return true;
	}

	// 加载MNIST测试集
	bool parse_test_data(std::string &data_path, std::vector<std::vector<float>> &test_images, std::vector<int> &test_labels,
		float min_val = -1.f, float max_val = 1.f,
		int padx = 0, int pady = 0)
	{
		bool state = parse_mnist_images(data_path + "/t10k-images-idx3-ubyte", &test_images, min_val, max_val, padx, pady)
			&& parse_mnist_labels(data_path + "/t10k-labels-idx1-ubyte", &test_labels);
		return state;
	}

	// 加载MNIST训练集
	bool parse_train_data(std::string &data_path, std::vector<std::vector<float>> &train_images, std::vector<int> &train_labels,
		float min_val = -1.f, float max_val = 1.f,
		int padx = 0, int pady = 0)
	{
		bool state = parse_mnist_images(data_path + "/train-images-idx3-ubyte", &train_images, min_val, max_val, padx, pady)
			&& parse_mnist_labels(data_path + "/train-labels-idx1-ubyte", &train_labels);
		return state;
	}
}
