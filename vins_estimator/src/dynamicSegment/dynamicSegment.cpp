#include "dynamicSegment.h"
#include "logger.h"

bool SampleOnnx::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
 
    if (!builder)
    {
        return false;
    }
 
 
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }
 
 
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
 
    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }
 
 
    //In this function will use the onnx file path
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }
 
 
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
 
    if (!mEngine)
    {
        return false;
    }
 
    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);
 
    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    std::cout<<"mOutputDims.nbDims:"<<mOutputDims.nbDims<<std::endl;
    ASSERT(mOutputDims.nbDims == 4);
 
    return true;
}

bool SampleOnnx::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                  SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
                                  SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                  SampleUniquePtr<nvonnxparser::IParser>& parser)
{
 
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str() , static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
 
    config->setMaxWorkspaceSize(512_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }
 
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
 
    return true;
}
 

bool SampleOnnx::infer(cv::Mat& img_raw, cv::Mat& infer_result)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
 
    //Read the image into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    

    // TicToc processInputTime;
    if (!processInput(buffers, img_raw))
    {
        return false;
    }
    // printf("   processInputTime: %f \n", processInputTime.toc());  //3.5ms
    // Memcpy from host input buffers to device input buffers
    // TicToc copyInputToDeviceTime;
    buffers.copyInputToDevice();
    // printf("copyInputToDeviceTime: %f \n", copyInputToDeviceTime.toc());
    // TicToc gpuInferTime;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    // printf("  gpuInferTime: %f \n", gpuInferTime.toc());  //1.5ms
    // Memcpy from device output buffers to host output buffers
    // TicToc copyOutputToHostTime;
    buffers.copyOutputToHost();
    // printf("copyOutputToHostTime: %f \n",copyOutputToHostTime.toc()); //0.5ms
    // TicToc getOutputTime;
    // Verify results
    if (!getOutput(buffers, infer_result))
    {
        return false;
    }
    // printf("getOutputTime: %f \n",getOutputTime.toc()); //0.3ms
    return true;
}
 
bool SampleOnnx::processInput(const samplesCommon::BufferManager& buffers, cv::Mat& img_raw)
{
    // std::cout << "mINputDims.d[0]:" << mInputDims.d[0] << std::endl;
    // std::cout << "mINputDims.d[1]:" << mInputDims.d[1] << std::endl;
    // std::cout << "mINputDims.d[2]:" << mInputDims.d[2] << std::endl;
    // std::cout << "mINputDims.d[3]:" << mInputDims.d[3] << std::endl;
 
    cv::Mat image_conv;
    cv::cvtColor(img_raw, image_conv, cv::COLOR_BGR2RGB);
    // std::cout << image_conv.channels() << "," << image_conv.size().width << "," << image_conv.size().height << std::endl;

    int target_size = 640;
    cv::resize(image_conv, image_conv, cv::Size(target_size, target_size),cv::INTER_LINEAR);


    static const float mean[3] = { 0.485f, 0.456f, 0.406f };
    static const float Std[3] = { 0.229f, 0.224f, 0.225f };

    const int channel = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];
    // Read a random digit file
    std::vector<float> fileData(inputH * inputW * channel);
    #pragma omp parallel for num_threads(20);
    for (int c = 0; c < channel; ++c)
    {
        for (int i = 0; i < image_conv.rows; ++i)
        {
            cv::Vec3b *p1 = image_conv.ptr<cv::Vec3b>(i);
            for (int j = 0; j < image_conv.cols; ++j)
            {
                fileData[c * image_conv.cols * image_conv.rows + i * image_conv.cols + j] = (p1[j][c] / 255.0f - mean[c]) / Std[c];
            }
        }
    }
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW * channel; i++)
    {
        hostDataBuffer[i] = fileData[i];
    }
    return true;
}

bool SampleOnnx::getOutput(const samplesCommon::BufferManager& buffers, cv::Mat& img_result)
{
    // const int outputSize = mOutputDims.d[1];
    // std::cout << "mOutputDims.d[0]: " << mOutputDims.d[0] << std::endl;
    // std::cout << "mOutputDims.d[1]: " << mOutputDims.d[1] << std::endl;
    // std::cout << "mOutputDims.d[2]: " << mOutputDims.d[2] << std::endl;
    // std::cout << "mOutputDims.d[3]: " << mOutputDims.d[3] << std::endl;
    // std::cout << "outputSize: " << outputSize << std::endl;
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    img_result = cv::Mat(640, 640, CV_8UC1, cv::Scalar(0));
    #pragma omp parallel for num_threads(20);
    for (int i = 0; i < 409600; i++)
    {
        if(output[i] < output[i+409600])
        {
            int row_number = i/640;
            int col_number = i % 640;
            img_result.at<int8_t>(row_number,col_number) = 255;
        }
    }
    // img_result = mask.clone();
    // cv::imshow("img",mask);
    // cv::waitKey(0);
    return true;
}
 