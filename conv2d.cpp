#define INCLUDE_PYTHON_INTERFACE 1
#define INCLUDE_PYBIND_PYTHON_INTERFACE 1

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "NumCpp.hpp"
#include <iostream>
#include <string>
#include<thread>
using namespace  std;
namespace py=pybind11;

template<typename T>
py::array_t<double> cpp_dot(py::array_t<T, py::array::c_style> inArray1, py::array_t<T, py::array::c_style> inArray2)
{
    auto array1 = nc::pybindInterface::pybind2nc(inArray1);
    auto array2 = nc::pybindInterface::pybind2nc(inArray2);
    auto dotProduct = nc::dot<double>(array1, array2);
    return nc::pybindInterface::nc2pybind(dotProduct);
}

template<typename T>
py::array_t<double> conv2d_pure(int s,string mode, py::array_t<T, py::array::c_style> kernel_py, py::array_t<T, py::array::c_style> input_py)
{
    auto kernel = nc::pybindInterface::pybind2nc(kernel_py);
    auto image = nc::pybindInterface::pybind2nc(input_py);
    int k=kernel.shape().rows;
    int h=image.shape().rows;
    int w=image.shape().cols;
    int p_h,p_w;
    if(mode=="same"){
        p_h=(s*(h-1)+k-h)/2;
        p_w=(s*(w-1)+k-w)/2;
    }
    else if(mode=="valid"){
        p_h=0;
        p_w=0;
    }
    else if(mode=="full"){
        p_h=k-1;
        p_w=k-1;
    }
    else{
        assert(false);
    }
    int out_h = (h + 2 * p_h - k) /s + 1;
    int out_w = (w + 2 * p_w - k) /s + 1;
    //填充后的image
    auto padded_img = nc::zeros<double>(h + 2 * p_h, w + 2 * p_w);
    padded_img.put(nc::Slice(p_h,p_h + h), nc::Slice(p_w,p_w + w),image) ;
    auto image_mat=nc::zeros<double>(out_h ,out_w);
//    nc::NdArray<double> shapedKernel=kernel.reshape(k*k,1);
    for(int i=0;i<out_h;++i){
        for(int j=0;j<out_w;++j){
            double sum=0;
            for(int x=i*s;x<i*s+k;++x){
                for(int y=j*s;y<j*s+k;++y){
                    sum+=static_cast<double>(padded_img(x,y)*kernel(x-i*s,y-j*s));
                }
            }
            image_mat.put(i,j,sum);
//            auto window=padded_img(nc::Slice(i * s,(i * s +k)),nc::Slice(j*s,j*s+k));
//            image_mat.put(i,j,nc::dot<double>(window.reshape(1,k*k),shapedKernel).item());
        }
    }
    return nc::pybindInterface::nc2pybind(image_mat);
}



template<typename T>
py::array_t<double> conv2d(int s,string mode, py::array_t<T, py::array::c_style> kernel_py, py::array_t<T, py::array::c_style> input_py)
{
    auto kernel = nc::pybindInterface::pybind2nc(kernel_py);
    auto image = nc::pybindInterface::pybind2nc(input_py);
    int k=kernel.shape().rows;
    int h=image.shape().rows;
    int w=image.shape().cols;
    int p_h,p_w;
    if(mode=="same"){
        p_h=(s*(h-1)+k-h)/2;
        p_w=(s*(w-1)+k-w)/2;
    }
    else if(mode=="valid"){
        p_h=0;
        p_w=0;
    }
    else if(mode=="full"){
        p_h=k-1;
        p_w=k-1;
    }
    else{
        assert(false);
    }
    int out_h = (h + 2 * p_h - k) /s + 1;
    int out_w = (w + 2 * p_w - k) /s + 1;
    //填充后的image
    auto padded_img = nc::zeros<double>(h + 2 * p_h, w + 2 * p_w);
    padded_img.put(nc::Slice(p_h,p_h + h), nc::Slice(p_w,p_w + w),image) ;
    auto image_mat=nc::zeros<double>(out_h * out_w, k*k);
    int row=0;
    nc::NdArray<double> window;
    for(int i=0;i<out_h;++i){
        for(int j=0;j<out_w;++j){
            window=padded_img(nc::Slice(i * s,(i * s +k)),nc::Slice(j*s,j*s+k));
            image_mat.put(row,image_mat.cSlice(),window.flatten());
            ++row;
        }
    }
    auto dotProduct = nc::dot<double>(image_mat, kernel.reshape(k*k,1)).reshape(out_h,out_w);
    return nc::pybindInterface::nc2pybind(dotProduct);
}


template<typename T>
void f(nc::NdArray<T>* image_mat,nc::NdArray<T> * padded_img,int out_h_start,int out_h,int out_w,int row,int s,int k){
    nc::NdArray<T> window;

    for(int i=out_h_start;i<out_h;++i){
        for(int j=0;j<out_w;++j){
            window=(*padded_img)(nc::Slice(i * s,(i * s +k)),nc::Slice(j*s,j*s+k));
            image_mat->put(row,image_mat->cSlice(),window.flatten());
            ++row;
        }
    }
}

template<typename T>
py::array_t<double> conv2d_multi(int s,string mode, py::array_t<T, py::array::c_style> kernel_py, py::array_t<T, py::array::c_style> input_py)
{
    auto kernel = nc::pybindInterface::pybind2nc(kernel_py);
    auto image = nc::pybindInterface::pybind2nc(input_py);
    int k=kernel.shape().rows;
    int h=image.shape().rows;
    int w=image.shape().cols;
    int p_h,p_w;
    if(mode=="same"){
        p_h=(s*(h-1)+k-h)/2;
        p_w=(s*(w-1)+k-w)/2;
    }
    else if(mode=="valid"){
        p_h=0;
        p_w=0;
    }
    else if(mode=="full"){
        p_h=k-1;
        p_w=k-1;
    }
    else{
        assert(false);
    }
    int out_h = (h + 2 * p_h - k) /s + 1;
    int out_w = (w + 2 * p_w - k) /s + 1;
    //填充后的image
    auto padded_img = nc::zeros<double>(h + 2 * p_h, w + 2 * p_w);
    padded_img.put(nc::Slice(p_h,p_h + h), nc::Slice(p_w,p_w + w),image) ;
    auto image_mat=nc::zeros<double>(out_h * out_w, k*k);

    thread t1(f<double>,&image_mat,&padded_img,0,out_h/2,out_w,0,s,k);
    thread t2(f<double>,&image_mat,&padded_img,out_h/2,out_h,out_w,out_h * out_w/2,s,k);
    t1.join();
    t2.join();
    auto dotProduct = nc::dot<double>(image_mat, kernel.reshape(k*k,1)).reshape(out_h,out_w);
    return nc::pybindInterface::nc2pybind(dotProduct);
}

PYBIND11_MODULE(example, m)
{
    m.doc() = "This is an example of using NumCpp with python and NumPy.";

    m.def("cpp_dot_double", &cpp_dot<double>,
          pybind11::arg("inArray1"),
          pybind11::arg("inArray2"),
          "Returns the dot project of the two arrays.");

    m.def("conv2d", &conv2d<double>,
          pybind11::arg("s"),
          pybind11::arg("mode"),
          pybind11::arg("kernel_py"),
          pybind11::arg("input_py"),
          "Returns the dot project of the two arrays.");

    m.def("conv2d_multi", &conv2d_multi<double>,
          pybind11::arg("s"),
          pybind11::arg("mode"),
          pybind11::arg("kernel_py"),
          pybind11::arg("input_py"),
          "Returns the dot project of the two arrays.");

    m.def("conv2d_pure", &conv2d_pure<double>,
          pybind11::arg("s"),
          pybind11::arg("mode"),
          pybind11::arg("kernel_py"),
          pybind11::arg("input_py"),
          "Returns the dot project of the two arrays.");
}

//#include <pybind11/pybind11.h>
//
//int add(int i, int j) {
//    return i + j;
//}
//
//PYBIND11_MODULE(example, m) {
//    m.doc() = "pybind11 example plugin"; // optional module docstring
//
//    m.def("add", &add, "A function which adds two numbers");
//}