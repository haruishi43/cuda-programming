#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
// #include <cuda_runtime.h>

namespace py = pybind11;
using namespace std;

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::Unaligned;

void convert_to_self(const cv::Mat &input, cv::Mat &output);

cv::Mat get_image(
    array<Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>, 3> &src)
{
    cv::Mat rgb[3];
    cv::Mat im_pano(src[0].rows(), src[0].cols(), CV_8UC3);
    for (int i=0; i<src.size(); i++)
    {
        cv::Mat channel(src[i].rows(), src[i].cols(), CV_8UC1, src[i].data());
        rgb[i] = channel;
    }
    cv::merge(rgb, 3, im_pano);
    
    cv::Mat output(im_pano.rows, im_pano.cols, CV_8UC3);
    convert_to_self(im_pano, output);
    
    return output;
}


PYBIND11_MODULE(extension, m)
{
    m.def("get_image", &get_image);

    // Cuffer protocol for return value
    py::class_<cv::Mat>(m, "Image", py::buffer_protocol())
        .def_buffer([](cv::Mat& im) -> py::buffer_info{
            return py::buffer_info(
                // pointer to buffer
                im.data,
                //size of one scalar
                sizeof(unsigned char),
                // Python struct-style format descriptor
                py::format_descriptor<unsigned char>::format(),
                // Number of dimensions
                3,
                // Buffer dimensions
                { im.rows, im.cols, im.channels() },
                // Strides (in bytes) for each index
                {
                    sizeof(unsigned char) * im.channels() * im.cols,
                    sizeof(unsigned char) * im.channels(),
                    sizeof(unsigned char)
                }
            );
        });
}
