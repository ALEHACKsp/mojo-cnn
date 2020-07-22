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
	// �������ݼ�����
	std::string data_name() { return std::string("MNIST"); }

	// �ֽ�ת��
	template<typename T = int>
	T* reverse_endian(T* p)
	{
		std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
		return p;
	}

	/// <summary>
	/// ����MNIST���ݼ��ı�ǩ.
	/// </summary>
	/// <param name="label_file">��ǩ�ļ�·��.</param>
	/// <param name="labels">���ؽ��������ı�ǩ.</param>
	/// <returns></returns>
	bool parse_mnist_labels(const std::string& label_file, std::vector<int> *labels)
	{
		//�Զ����Ʒ�ʽ��ȡ�ļ�
		std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
		if (ifs.is_open() == false) return false;

		//��ȡ��־λ�ͱ�ǩ����
		int magic_number = 0, num_items = 0;
		ifs.read((char*)&magic_number, 4);
		ifs.read((char*)&num_items, 4);

		//�����ֽ�ת��
		reverse_endian(&magic_number);
		reverse_endian(&num_items);

		//��ȡ��ǩ����
		for (int i = 0; i < num_items; i++)
		{
			unsigned char label = 0;
			ifs.read((char*)&label, 1);
			labels->push_back((int)label);
		}

		//�رձ�ǩ�ļ�
		ifs.close();

		//����
		return true;
	}

	// MNIST���ݼ�ͷ�ṹ
	struct mnist_header
	{
		int magic_number;	//��ʶ��
		int num_items;			//ͼ������
		int num_rows;			//ͼ����
		int num_cols;			//ͼ��߶�
	};

	/// <summary>
	/// ����MNISTͼ������.
	/// </summary>
	/// <param name="image_file">MNISTͼ���ļ�·��.</param>
	/// <param name="images">���ض�ȡ����ͼ������.</param>
	/// <param name="scale_min">��С����.</param>
	/// <param name="scale_max">�������.</param>
	/// <param name="x_padding">X�����.</param>
	/// <param name="y_padding">Y�����.</param>
	/// <returns></returns>
	bool parse_mnist_images(const std::string& image_file,
		std::vector<std::vector<float>> *images,
		float scale_min = -1.0, float scale_max = 1.0,
		int x_padding = 0, int y_padding = 0)
	{
		//�Զ����Ʒ�ʽ��ͼ���ļ�
		std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
		if (ifs.is_open() == false) return false;

		mnist_header header{ 0 };

		// ��ȡ�ļ�ͷ��Ϣ
		ifs.read((char*)&header.magic_number, 4);
		ifs.read((char*)&header.num_items, 4);
		ifs.read((char*)&header.num_rows, 4);
		ifs.read((char*)&header.num_cols, 4);

		// �ֽ�ת��
		reverse_endian(&header.magic_number);
		reverse_endian(&header.num_items);
		reverse_endian(&header.num_rows);
		reverse_endian(&header.num_cols);

		/*
		Ϊʲô��Ҫ����2��?
		��Ϊͼ������ߺ��ұ�ѽ,�ϱߺ��±�ѽ,����ֻ�����߶�������ұ߰�?
		*/
		const int width = header.num_cols + 2 * x_padding;
		const int height = header.num_rows + 2 * y_padding;

		// ��ȡÿһ��ͼ��
		for (int i = 0; i < header.num_items; i++)
		{
			// ��ȡԭʼͼ������
			std::vector<unsigned char> image_vec(header.num_rows * header.num_cols);
			ifs.read((char*)&image_vec[0], header.num_rows * header.num_cols);

			// ��Ź�һ����ͼ������
			std::vector<float> image;
			image.resize(width * height, scale_min);

			// ͼ����
			for (int y = 0; y < header.num_rows; y++)
			{
				for (int x = 0; x < header.num_cols; x++)
					image[width * (y + y_padding) + x + x_padding] =
					(image_vec[y * header.num_cols + x] / 255.0f) * //���ŵ�(0,1)֮��
					(scale_max - scale_min) +
					scale_min;
			}

			// ����ͼ��
			images->push_back(std::move(image));
		}

		//�ر��ļ�
		ifs.close();

		//����
		return true;
	}

	// ����MNIST���Լ�
	bool parse_test_data(std::string &data_path, std::vector<std::vector<float>> &test_images, std::vector<int> &test_labels,
		float min_val = -1.f, float max_val = 1.f,
		int padx = 0, int pady = 0)
	{
		bool state = parse_mnist_images(data_path + "/t10k-images-idx3-ubyte", &test_images, min_val, max_val, padx, pady)
			&& parse_mnist_labels(data_path + "/t10k-labels-idx1-ubyte", &test_labels);
		return state;
	}

	// ����MNISTѵ����
	bool parse_train_data(std::string &data_path, std::vector<std::vector<float>> &train_images, std::vector<int> &train_labels,
		float min_val = -1.f, float max_val = 1.f,
		int padx = 0, int pady = 0)
	{
		bool state = parse_mnist_images(data_path + "/train-images-idx3-ubyte", &train_images, min_val, max_val, padx, pady)
			&& parse_mnist_labels(data_path + "/train-labels-idx1-ubyte", &train_labels);
		return state;
	}
}
