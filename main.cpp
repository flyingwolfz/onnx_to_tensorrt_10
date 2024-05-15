#include "NvInfer.h"  
#include<iostream>
#include <cuda_runtime.h>  
#include "NvOnnxParser.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <chrono>

#include <helper_cuda.h>  //cuda samples/common

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace nvinfer1;  
using namespace nvonnxparser; 
using namespace std;
using namespace cv;

const char* inputName = "input"; 
const char* outputName = "output"; 

const int H = 1472;
const int W = 3072;

const int Hinput = H;
const int Winput = W;
const int channel = 3;
const int inputSize = channel * Hinput * Winput;
const int outputSize = channel * Hinput * Winput;
//const int outputSize = channel * H * W;

const char* onnxPath = "onnx.onnx";
const char* enginePath = "trt.trt";
bool forceBuild = false;     // rebuild engine if engine file already exists
const char* inputImagePath= "scene_116_gt.png";   //color
const char* outputImagePath = "out.png";   //color

extern "C"  void networkOutputToPhase(float* networkOutput, int W, int H, int C);

class Logger :public ILogger 
{
    void log(Severity severity, const char* msg) noexcept override 
    {
            if (severity < Severity::kWARNING)
                std::cout << msg << std::endl;
    };
}logger;

void printNetworkInfo(INetworkDefinition* network)
{
    cout << "-------printNetworkInfo------- "<< std::endl;
    auto inputNum = network->getNbInputs();// assume 1
    nvinfer1::Dims inDims = network->getInput(0)->getDimensions();// assume 4
    auto inputDim = inDims.nbDims;
    assert(inputNum ==1);
    assert(inputDim == 4);
    cout << "input num: " << inputNum << std::endl;  
    cout << "input dim: " << inputDim << std::endl;  
    cout << "input size: " << inDims.d[0]<<"," << inDims.d[1] << "," 
        << inDims.d[2] << "," << inDims.d[3] << std::endl;
    
    auto outputNum = network->getNbOutputs();// assume 1
    nvinfer1::Dims outDims = network->getOutput(0)->getDimensions();// assume 4
    auto outputDim = outDims.nbDims;
    assert(outputNum == 1);
    assert(outputDim == 4);
    cout << "output num: " << outputNum << std::endl; 
    cout << "output dim: " << outputDim << std::endl; 
    cout << "output size: " << outDims.d[0] << "," << outDims.d[1] << ","
        << outDims.d[2] << "," << outDims.d[3] << std::endl;
    
    //cout << "-------------------------- " << std::endl;

}

bool buildEngine(const char* onnx_path, const char* engine_path)
{
   
    IBuilder* builder = createInferBuilder(logger);
    assert(builder != nullptr);

    INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);

    IParser* parser = createParser(*network, logger);
    assert(parser != nullptr);

    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);

    // set config
    config->setFlag(BuilderFlag::kFP16);
    //config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 48);
    //config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);

    cout << "Loading ONNX file from: " << onnx_path << endl;
    if (parser->parseFromFile(onnx_path, static_cast<int32_t>(ILogger::Severity::kWARNING)))
    {
        cout << "ONNX parsed successfully" << endl;
    }
    else
    {
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    cout << "converting onnx to  engine, this may take a while........ " << endl;
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    assert(serializedModel != nullptr);
    cout << "successfully  convert onnx to  engine " << std::endl;

    std::ofstream a(engine_path, std::ios::binary);
    a.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    cout << "engine saved: " << engine_path << endl;

    printNetworkInfo(network);

    delete serializedModel;
    delete parser;
    delete network;
    delete config;
    delete builder;
    return true;
}

bool loadEngineFile(char*& readEngine, size_t& size)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        readEngine = new char[size];
        file.read(readEngine, size);
        file.close();
        cout << "engine file read" << endl;
        return true;
    }
    else
        return false;
}

void printEngineInfo(ICudaEngine* engine)
{
    cout << "-------printEngineInfo------- " << std::endl;
    auto numIOTensors = engine->getNbIOTensors();
    cout << "numbers of the IOTensors: " << numIOTensors <<endl;
    assert(numIOTensors == 2);  //in out

    auto inputName = engine->getIOTensorName(0);
    auto inputShape = engine->getTensorShape(inputName);
    cout << "inputname: " << inputName << std::endl;;
    auto inputDim = inputShape.nbDims;
    assert(inputDim == 4);
    cout << "input dim: " << inputDim << std::endl;
    cout << "input size: " << inputShape.d[0] << "," << inputShape.d[1] 
        << "," << inputShape.d[2] << "," << inputShape.d[3] << std::endl;

    auto outputName = engine->getIOTensorName(1);
    auto outputShape = engine->getTensorShape(outputName);
    cout << "outputname: " << outputName << std::endl;;
    auto outputDim = outputShape.nbDims;
    assert(outputDim == 4);
    cout << "output dim: " << outputDim << std::endl;
    cout << "output size: " << outputShape.d[0] << "," << outputShape.d[1] 
        << "," << outputShape.d[2] << "," << outputShape.d[3] << std::endl;

}


void loadImage(float* inputHost)
{

    Mat image = imread(inputImagePath);
     
    for (int y = 0; y < Hinput; ++y)
    {
        for (int x = 0; x < Winput; ++x)
        {
            Vec3b pixel = image.at<Vec3b>(y, x);
            inputHost[(0 * Hinput + y) * Winput + x] = pixel[0] / 255.0f; // B
            inputHost[(1 * Hinput + y) * Winput + x] = pixel[1] / 255.0f; // G
            inputHost[(2 * Hinput + y) * Winput + x] = pixel[2] / 255.0f; // R
        }
    }

}


void saveImage(float* outputHost)
{
    Mat out(H, W, CV_8UC3, 1);
    //Mat out(H, W, CV_8UC1, 1);

    for (int y = 0; y < H; ++y)
    {

        for (int x = 0; x < W; ++x)
        {
            //out.at<uchar>(y, x) = floor(outputHost[y * W + x] * 255.0);
            out.at<Vec3b>(y, x)[0] = floor(outputHost[(0 * H + y) * W + x] * 255.0);
            out.at<Vec3b>(y, x)[1] = floor(outputHost[(1 * H + y) * W + x] * 255.0);
            out.at<Vec3b>(y, x)[2] = floor(outputHost[(2 * H + y) * W + x] * 255.0);
        }
    }

    imwrite("1.png", out);

}

bool inference()
{
    cout << "-------inference info------- " << std::endl;

    char* readEngine = nullptr;
    size_t size=0;
    assert(loadEngineFile(readEngine, size) == true);

    IRuntime* runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(readEngine, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] readEngine;

    printEngineInfo(engine);
    
    //float* inputHost = (float*)malloc(inputSize * sizeof(float));
    //float* outputHost = (float*)malloc(outputSize * sizeof(float));
    float* inputHost;
    checkCudaErrors(cudaMallocHost(&inputHost, inputSize * sizeof(float)));
    float* outputHost;
    checkCudaErrors(cudaMallocHost(&outputHost, outputSize * sizeof(float)));
    assert(inputHost != nullptr);
    assert(outputHost != nullptr);

    loadImage(inputHost);

    float* inputDevice;
    float* outputDevice;
    checkCudaErrors(cudaMalloc(&inputDevice, inputSize * sizeof(float)));
    checkCudaErrors(cudaMalloc(&outputDevice, outputSize * sizeof(float)));
    assert(inputDevice != nullptr);
    assert(outputDevice != nullptr);

    bool status;
    status = context->setTensorAddress(inputName, inputDevice);
    assert(status != false);
    status = context->setTensorAddress(outputName, outputDevice);
    assert(status != false);

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cudaMemcpyAsync(inputDevice, inputHost, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    //warmUpKernel();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int y = 0; y <100; ++y)
    {
        //checkCudaErrors(cudaMemcpyAsync(inputDevice, inputHost, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));
        //checkCudaErrors(cudaStreamSynchronize(stream));

        status = context->enqueueV3(stream);
        assert(status != false);
        checkCudaErrors(cudaStreamSynchronize(stream));

        //checkCudaErrors(cudaMemcpyAsync(outputHost, outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        //checkCudaErrors(cudaStreamSynchronize(stream));
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Time cost: " << elapsedTime/100.0 << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    networkOutputToPhase(outputDevice,  W,  H, channel);

    checkCudaErrors(cudaMemcpyAsync(outputHost, outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    saveImage(outputHost);

    checkCudaErrors(cudaFree(inputDevice));
    checkCudaErrors(cudaFree(outputDevice));
    checkCudaErrors(cudaStreamDestroy(stream));
    //free(inputHost);
    //free(outputHost);
    checkCudaErrors(cudaFreeHost(outputHost));
    checkCudaErrors(cudaFreeHost(inputHost));
    delete context;
    delete engine;
    delete runtime;
    return true;
}



bool isFileExists(const std::string& filename) 
{
    std::ifstream file(filename);
    return file.good();
}

bool getEngine()
{

    cout << "-------getEngine info------- " << std::endl;
    if (forceBuild)
    {
        cout << "usd force_build, build engine..." << endl;
        return buildEngine(onnxPath, enginePath);
    }
    else
    {
        if (isFileExists(enginePath))
        {
            cout << "engine exists, use existing engine..." << endl;
            return true;
        }
        else
        {
            cout << "engine doesn't exist, build engine..." << endl;
            return buildEngine(onnxPath, enginePath);
        }
    }
}

int main()
{

    cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
 
    assert(getEngine() == true);
    //test();

    assert(inference() == true);

    return 0;
}