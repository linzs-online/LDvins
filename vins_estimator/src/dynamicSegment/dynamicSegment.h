#pragma once
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "../utility/tic_toc.h"
class SampleOnnx
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
public:
    SampleOnnx(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
        
    }
    bool build();
    bool infer(cv::Mat& img_raw, cv::Mat& infer_result);
    bool getOutput(const samplesCommon::BufferManager& buffers, cv::Mat& img_result);
private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
                          SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvonnxparser::IParser>& parser);
    bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat& img_raw);
};

class DynamicSegment
{
private:
    /* data */
    cv::Mat imageRaw;
    cv::Mat mask;
public:
    DynamicSegment(cv::Mat img_raw):imageRaw(img_raw) {
        mask = cv::Mat(imageRaw.rows, imageRaw.rows, CV_8UC1, cv::Scalar(255));
    }

    cv::Mat getSegmentResult(cv::Mat& img_raw) {
        return mask;
    }

    ~DynamicSegment() {

    }
};
