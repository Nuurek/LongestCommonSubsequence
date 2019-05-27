#include <iostream>
#include <fstream>
#include <string>
#include <streambuf>
#include <set>
#include <algorithm>
#include <CL/cl.hpp>

int main()
{
    std::string mode, x, y;
    std::cout << "Do you want to use test data? y/n\n";
    std::getline(std::cin, mode);
    if (mode == "y") {
        x = "1232412";
        y = "243121";
    } else {
        std::cout << "First sequence: ";
        std::getline(std::cin, x);
        std::cout << "Second sequence: ";
        std::getline(std::cin, y);
    }

    std::cout << "Searching for a Longest Common Subsequence (LCS) between ";
    std::cout << x << " and " << y << "\n";

    unsigned long lcsWidth = x.length() + 1;
    unsigned long lcsHeight = y.length() + 1;
    unsigned long lcsSize = lcsWidth * lcsHeight;
    unsigned long numberOfWorkItems = std::max(lcsWidth, lcsHeight);

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
            std::string("calculateLCS.cl")
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
    const auto bufferSize = lcsSize * sizeof(unsigned long);
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

    auto hostLCSBuffer = std::vector<unsigned long>(lcsSize);
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

    std::vector<std::vector<unsigned long>> lcs;
    for (unsigned long i = 0; i < lcsWidth; i++) {
        lcs.emplace_back(std::vector<unsigned long>(lcsHeight));

        for (unsigned long j = 0; j < lcsHeight; j++) {
            lcs[i][j] = hostLCSBuffer[j * lcsWidth + i];
            std::cout << lcs[i][j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::vector<std::vector<std::set<std::string>>> results;
    for (unsigned long i = 0; i < lcsWidth; i++) {
        results.emplace_back(std::vector<std::set<std::string>>(lcsHeight));
    }

    const size_t numberOfIterations = lcsWidth + lcsHeight;

    for (unsigned long n = 0; n < numberOfIterations; n++) {
        const unsigned long start = n < lcsHeight ? 0 : n - lcsHeight + 1;
        const unsigned long end = n < lcsWidth ? n : (lcsWidth - 1);
        for (unsigned long i = start; i <= end; i++) {
            const unsigned long j = n - i;

            std::set<std::string> result;
            if (i == 0 || j == 0) {
                result = std::set<std::string>();
                result.insert(std::string());
            } else if (x[i - 1] != y[j - 1]) {
                auto set1 = std::set<std::string>();
                auto set2 = std::set<std::string>();

                if (lcs[i - 1][j] >= lcs[i][j - 1]) {
                    set1 = results[i - 1][j];
                }

                if (lcs[i - 1][j] <= lcs[i][j - 1]) {
                    set2 = results[i][j - 1];
                }

                set1.insert(set2.begin(), set2.end());
                result = set1;
            } else {
                result = std::set<std::string>();

                const auto& previousResult = results[i - 1][j - 1];
                const auto character = x[i - 1];
                for (const auto& sequence : previousResult) {
                    auto extendedSequence = sequence;
                    extendedSequence += character;
                    result.insert(extendedSequence);
                }
            }

            results[i][j] = result;

            std::cout << "[" << i << "][" << j << "] ";
            for (const auto& sequence : result) {
                std::cout << "[" << (!sequence.empty() ? sequence : "EMPTY SEQUENCE") << "], ";
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";

    const auto& finalResult = results[lcsWidth - 1][lcsHeight - 1];
    if (!finalResult.empty()) {
        std::cout << "Found " << finalResult.size() << " longest common subsequences:\n";
        for (const auto& sequence : finalResult) {
            std::cout << "[" << (!sequence.empty() ? sequence : "EMPTY SEQUENCE") << "]\n";
        }
    } else {
        std::cout << "Nothing found\n";
    }

    clFlush(commandQueue);
    clFinish(commandQueue);
    clReleaseKernel(calculateLCSKernel);
    clReleaseProgram(program);
    clReleaseMemObject(lcsBuffer);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return 0;
}