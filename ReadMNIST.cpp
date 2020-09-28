#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include "ReadMNIST.h"

// Miscellaneous function
int reverseInt(int iSample)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = iSample & 255;
    ch2 = (iSample >> 8) & 255;
    ch3 = (iSample >> 16) & 255;
    ch4 = (iSample >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

std::vector<cv::Mat> readMnistImages(std::string setFullPath)
{
    std::vector<cv::Mat> arr;
    std::ifstream file(setFullPath, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        std::cout << "Dataset size       : " << number_of_images << "\n";  
        
        for (int i=0; i<number_of_images; ++i){    
            cv::Mat im(n_rows, n_cols, CV_8UC1);    
            for(int r=0; r<n_rows; ++r){
                for(int c=0; c<n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char *)&temp, sizeof(temp));
                    im.at<uint8_t> (r,c) = (uint8_t) temp;
                }
            }

            arr.push_back(im);
            im.release();
        }
    }
    return arr;
}

// Return a column containing the labels per image
std::vector<int> readMnistLabels(std::string setPath, int numOfLabels)
{
    std::vector<int> vecLabel(numOfLabels);

    std::ifstream file(setPath, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int numOfLabels = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = reverseInt(numOfLabels);

        for (int iSample = 0; iSample < numOfLabels; ++iSample)
        {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            vecLabel[iSample] = (int) temp;
        }
    }
    return vecLabel;
}