#include <iostream>
#include <fstream>
#include <string>
#include <streambuf>
#include <CL/cl.hpp>

int main()
{
    std::string mode, x, y;
    std::cout << "Do you want to use test data? y/n\n";
    std::cin >> mode;
    if (mode == "y") {
        x = "1232412";
        y = "243121";
    } else {
        std::cout << "First sequence: ";
        std::cin >> x;
        std::cout << "Second sequence: ";
        std::cin >> y;
    }

    std::cout << "Searching for a Longest Common Subsequence (LCS) between ";
    std::cout << x << " and " << y << "\n";

    size_t lcsWidth = x.length() + 1;
    size_t lcsHeight = y.length() + 1;
    size_t lcsSize = lcsWidth * lcsHeight;
    size_t numberOfWorkItems = std::max(lcsWidth, lcsHeight);

    cl_uint DESIRED_NUMBER_OF_PLATFORMS = 1;
    cl_platform_id platformId;
    cl_uint numberOfPlatforms;
    const cl_int getPlatformIDsResult = clGetPlatformIDs(
            DESIRED_NUMBER_OF_PLATFORMS,
            &platformId,
            &numberOfPlatforms
    );
    if (getPlatformIDsResult != CL_SUCCESS) {
        exit(1);
    }

    const size_t PLATFORM_NAME_SIZE = 128;
    char platformName[PLATFORM_NAME_SIZE];
    size_t platformNameSize;
    const cl_int getPlatformInfoResult = clGetPlatformInfo(
            platformId,
            CL_PLATFORM_NAME,
            PLATFORM_NAME_SIZE,
            &platformName,
            &platformNameSize
    );
    std::cout << "Using " << std::string(platformName) << ", ";

    cl_uint DESIRED_NUMBER_OF_DEVICES = 1;
    cl_device_id deviceId;
    cl_uint numberOfDevices;
    const cl_int getDeviceIDsResult = clGetDeviceIDs(
            platformId,
            CL_DEVICE_TYPE_GPU,
            DESIRED_NUMBER_OF_DEVICES,
            &deviceId,
            &numberOfDevices
    );
    if (getDeviceIDsResult != CL_SUCCESS) {
        exit(1);
    }

    const size_t DEVICE_NAME_SIZE = 128;
    char deviceName[DEVICE_NAME_SIZE ];
    size_t deviceNameSize;
    const cl_int getDeviceInfoResult = clGetDeviceInfo(
            deviceId,
            CL_DEVICE_NAME,
            DEVICE_NAME_SIZE,
            &deviceName,
            &deviceNameSize
    );
    std::cout << std::string(deviceName) << "\n";

    cl_int createContextResult;
    cl_context context = clCreateContext(
            nullptr,
            DESIRED_NUMBER_OF_DEVICES,
            &deviceId,
            nullptr,
            nullptr,
            &createContextResult
    );

    cl_int createQueueResult;
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(
            context,
            deviceId,
            nullptr,
            &createQueueResult
    );

    const auto kernelsFileNames = {
            std::string("calculateLCS.cl"),
            std::string("traverseLCS.cl")
    };
    auto kernelsSources = std::vector<std::string>();

    std::string kernelSourceLine;
    for (const auto& kernelFileName : kernelsFileNames) {
        std::ifstream kernelFile(kernelFileName);
        if (kernelFile.is_open()) {
            const auto kernelSource = std::string(
                    std::istreambuf_iterator<char>(kernelFile),
                    std::istreambuf_iterator<char>()
            );
            kernelsSources.push_back(kernelSource);
        }
    }

    const auto numberOfKernels = (cl_uint)kernelsSources.size();
    auto kernelStrings = std::vector<const char*>();
    auto kernelLengths = std::vector<size_t>();
    for (const auto& kernel : kernelsSources) {
        kernelStrings.push_back(kernel.data());
        kernelLengths.push_back(kernel.length());
    }
    cl_int createProgramResult;
    const auto program = clCreateProgramWithSource(
            context,
            numberOfKernels,
            kernelStrings.data(),
            kernelLengths.data(),
            &createProgramResult
    );

    const auto buildProgramResult = clBuildProgram(
            program,
            DESIRED_NUMBER_OF_DEVICES,
            &deviceId,
            nullptr,
            nullptr,
            nullptr
    );
    if (buildProgramResult == CL_SUCCESS) {
        std::cout << "Successfully compiled all kernels\n";
    } else {
        std::string log;
        log.resize(1024);
        size_t resultSize;
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 1024, (void*)log.data(), &resultSize);
        log.resize(resultSize);
        std::cout << log << "\n";
        exit(1);
    }

    cl_int createKernelResult;
    const auto calculateLCSKernel = clCreateKernel(
            program,
            "calculateLCS",
            &createKernelResult
    );

    cl_int createBufferResult;
    const auto bufferSize = lcsSize * sizeof(unsigned int);
    auto lcsBuffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            bufferSize,
            nullptr,
            &createBufferResult
    );
    auto xBuffer = clCreateBuffer(
            context,
            CL_MEM_COPY_HOST_PTR,
            x.length() * sizeof(char),
            (void*)x.data(),
            &createBufferResult
    );
    auto yBuffer = clCreateBuffer(
            context,
            CL_MEM_COPY_HOST_PTR,
            y.length() * sizeof(char),
            (void*)y.data(),
            &createBufferResult
    );

    clSetKernelArg(calculateLCSKernel, 0, sizeof(cl_mem), &lcsBuffer);
    clSetKernelArg(calculateLCSKernel, 1, sizeof(cl_mem), &xBuffer);
    clSetKernelArg(calculateLCSKernel, 2, sizeof(lcsWidth), &lcsWidth);
    clSetKernelArg(calculateLCSKernel, 3, sizeof(cl_mem), &yBuffer);
    clSetKernelArg(calculateLCSKernel, 4, sizeof(lcsHeight), &lcsHeight);

    size_t globalWorkSize[] = { numberOfWorkItems, 0, 0 };
    clEnqueueNDRangeKernel(
            commandQueue,
            calculateLCSKernel,
            1,
            nullptr,
            globalWorkSize,
            nullptr,
            0,
            nullptr,
            nullptr
    );

    auto hostLCSBuffer = std::vector<unsigned int>(lcsSize);
    clEnqueueReadBuffer(
            commandQueue,
            lcsBuffer,
            CL_TRUE,
            0,
            bufferSize,
            hostLCSBuffer.data(),
            0,
            nullptr,
            nullptr
    );

    for (unsigned int j = 0; j < lcsHeight; j++) {
        for (unsigned int i = 0; i < lcsWidth; i++) {
            std::cout << hostLCSBuffer[j * lcsWidth + i] << "\t";
        }
        std::cout << "\n";
    }

    const auto traverseLCSKernel = clCreateKernel(
            program,
            "traverseLCS",
            &createKernelResult
    );

    const unsigned int resultStringLength = hostLCSBuffer[lcsSize - 1];
    const size_t resultCStringSize = sizeof(char) * (resultStringLength + 1);
    const size_t resultCellSize = sizeof(unsigned int) + resultStringLength * resultStringLength * resultCStringSize;
    const size_t resultsSize = lcsSize * resultCellSize;
    auto resultsBuffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            resultsSize,
            nullptr,
            &createBufferResult
    );

    clSetKernelArg(traverseLCSKernel, 0, sizeof(cl_mem), &lcsBuffer);
    clSetKernelArg(traverseLCSKernel, 1, sizeof(cl_mem), &xBuffer);
    clSetKernelArg(traverseLCSKernel, 2, sizeof(lcsWidth), &lcsWidth);
    clSetKernelArg(traverseLCSKernel, 3, sizeof(cl_mem), &yBuffer);
    clSetKernelArg(traverseLCSKernel, 4, sizeof(lcsHeight), &lcsHeight);
    clSetKernelArg(traverseLCSKernel, 5, sizeof(cl_mem), &resultsBuffer);

    clEnqueueNDRangeKernel(
            commandQueue,
            traverseLCSKernel,
            1,
            nullptr,
            globalWorkSize,
            nullptr,
            0,
            nullptr,
            nullptr
    );

    auto hostResultsBuffer = std::vector<char>(resultsSize);
    clEnqueueReadBuffer(
            commandQueue,
            resultsBuffer,
            CL_TRUE,
            0,
            resultsSize,
            hostResultsBuffer.data(),
            0,
            nullptr,
            nullptr
    );

    for (unsigned int j = 0; j < lcsHeight; j++) {
        for (unsigned int i = 0; i < lcsWidth; i++) {
            const unsigned int numberOfStrings = hostResultsBuffer[(j * lcsWidth + i) * resultCellSize];
            std::cout << "[" << i << "][" << j << "] " << numberOfStrings << "\n";
        }
    }

    clFlush(commandQueue);
    clFinish(commandQueue);
    clReleaseKernel(calculateLCSKernel);
    clReleaseKernel(traverseLCSKernel);
    clReleaseProgram(program);
    clReleaseMemObject(lcsBuffer);
    clReleaseMemObject(resultsBuffer);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return 0;
}