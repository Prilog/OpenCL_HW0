//Device should support 16x16 work group size
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

constexpr auto MAX_SOURCE_SIZE = (0x100000);
#define TILE_W 16
#define TILE_H 16

struct data {
    float* value;
    size_t size;

    data(size_t sz) {
        size = sz;
        value = (float*)malloc(sizeof(float) * sz);
    }

    ~data() {
        free(value);
    }

    void fill() {
        for (size_t i = 0; i < size; i++) {
            value[i] = float(rand() % 10);
        }
    }

    float& operator[](size_t ind) {
        return value[ind];
    }
};

// Load the kernel source code into the array
void load_kernel_source(const char* file_name, char* source, size_t* length) {
    printf("Loading kernel\n");
    FILE* fp = fopen(file_name, "r");
    if (!fp) {
        printf("Failed to load kernel.\n");
        exit(1);
    }
    *length = fread(source, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    printf("kernel loading done\n");
}

struct device {
    bool failed;
    cl_device_id id;
    char* name;
    cl_device_type type;
    size_t max_work_group_size;
    cl_uint max_work_item_dimensions;
    size_t* max_work_item_sizes;
    cl_bool unified_memory;

    device(cl_device_id d_id) {
        failed = false;
        id = d_id;
        name = (char*)malloc(1024 * sizeof(char));
        max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * 3);
        cl_int ret = 0;
        ret = clGetDeviceInfo(id, CL_DEVICE_NAME, 1024, name, NULL);
        printf("ret at %d is %d\n", __LINE__, ret);
        if (ret) {
            failed = true;
            return;
        }
        ret = clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
        printf("ret at %d is %d\n", __LINE__, ret);
        if (ret) {
            failed = true;
            return;
        }
        ret = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
        printf("ret at %d is %d\n", __LINE__, ret);
        if (ret) {
            failed = true;
            return;
        }
        ret = clGetDeviceInfo(id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL);
        printf("ret at %d is %d\n", __LINE__, ret);
        if (ret) {
            failed = true;
            return;
        }
        ret = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, NULL);
        printf("ret at %d is %d\n", __LINE__, ret);
        if (ret) {
            failed = true;
            return;
        }
        if (max_work_item_dimensions != 3) {
            free(max_work_item_sizes);
            max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * max_work_item_dimensions);
        }
        ret = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL);
        printf("ret at %d is %d\n", __LINE__, ret);
        if (ret) {
            failed = true;
            return;
        }
    }

    ~device() {
        free(name);
        free(max_work_item_sizes);
        clReleaseDevice(id);
    }

    void print_info() {
        printf("Device Name: %s\n", name);
        printf("Device Type: %s\n", (type == CL_DEVICE_TYPE_GPU ? "GPU" : "Other"));
        printf("Unified Memory: %s\n", (unified_memory ? "True" : "False"));
        printf("Max Work Group Size: %zu\n", max_work_group_size);
        printf("Max Work Item Dimensions: %ld\n", max_work_item_dimensions);
        printf("Max Work Item Sizes: ");
        for (size_t i = 0; i < max_work_item_dimensions; i++) {
            printf("%zu ", max_work_item_sizes[i]);
        }
        printf("\n");
    }
};

int main(void) {
    printf("started running\n");
    //
    printf("Extracting source\n");
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size;
    load_kernel_source("kernel.cl", source_str, &source_size);
    // Get platform and device information
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_device_id chosen_device = NULL;
    size_t max_work_group_size;
    bool is_gpu = false;
    bool is_unified = false;

    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Couldn't get platforms amount\n");
        return ret;
    }
    cl_platform_id* platforms;
    platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

    ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Couldn't get platform IDs\n");
        return ret;
    }

    for (size_t i = 0; i < ret_num_platforms; i++) {
        const size_t device_cap = 64;
        cl_device_id devices[device_cap];

        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_cap,
            devices, &ret_num_devices);
        printf("ret at %d is %d\n", __LINE__, ret);
        if (ret) {
            printf("Couldn't get devices\n");
            continue;
        }

        for (size_t j = 0; j < ret_num_devices; j++) {
            cl_device_id cur_device = devices[j];
            device info(cur_device);
            if (info.failed) {
                printf("Couldn't get device's info\n");
                continue;
            }
            info.print_info();
            if (chosen_device == NULL) {
                chosen_device = cur_device;
                max_work_group_size = info.max_work_group_size;
                is_gpu = info.type == CL_DEVICE_TYPE_GPU;
                is_unified = info.unified_memory;
                continue;
            }
            if (!is_gpu && info.type == CL_DEVICE_TYPE_GPU) {
                chosen_device = cur_device;
                max_work_group_size = info.max_work_group_size;
                is_gpu = info.type == CL_DEVICE_TYPE_GPU;
                is_unified = info.unified_memory;
                continue;
            }
            if (is_gpu == (info.type == CL_DEVICE_TYPE_GPU) && is_unified && !info.unified_memory) {
                chosen_device = cur_device;
                max_work_group_size = info.max_work_group_size;
                is_gpu = info.type == CL_DEVICE_TYPE_GPU;
                is_unified = info.unified_memory;
                continue;
            }
            if (is_gpu == (info.type == CL_DEVICE_TYPE_GPU) && is_unified == info.unified_memory && max_work_group_size < info.max_work_group_size) {
                chosen_device = cur_device;
                max_work_group_size = info.max_work_group_size;
                is_gpu = info.type == CL_DEVICE_TYPE_GPU;
                is_unified = info.unified_memory;
            }
        }
    }
    device chosen_device_info(chosen_device);
    if (chosen_device_info.failed) {
        printf("Failed to get info of chosen device\n");
        return 0;
    }
    printf("Chosen device: %s\n", chosen_device_info.name);
    free(platforms);
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &chosen_device, NULL, NULL, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to create device's context\n");
        return ret;
    }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, chosen_device, CL_QUEUE_PROFILING_ENABLE, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to create command queue\n");
        return ret;
    }

    // Create the two input vectors
    printf("Initialising matrices\n");
    const unsigned int N = 2048;
    const unsigned int K = 512;
    const unsigned int M = 1024;
    data a(N * K);
    a.fill();
    data b(K * M);
    b.fill();
    data c(N * M);
    data ac(N * M);
    printf("Processing sample answer\n");
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            c[i * M + j] = 0;
            ac[i * M + j] = 0;
            for (size_t k = 0; k < K; k++) {
                ac[i * M + j] += a[i * K + k] * b[k * M + j];
            }
            //printf("ac[%d][%d] = %f\n", i, j, ac[i][j]);
        }
    }
    printf("Initialisation completed\n");

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * N * K, NULL, &ret);
    if (ret) {
        printf("Failed to create memory buffer\n");
        return ret;
    }
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * K * M, NULL, &ret);
    if (ret) {
        printf("Failed to create memory buffer\n");
        return ret;
    }
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * N * M, NULL, &ret);
    if (ret) {
        printf("Failed to create memory buffer\n");
        return ret;
    }

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        sizeof(float) * N * K, a.value, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to copy array to buffer\n");
        return ret;
    }

    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        sizeof(float) * K * M, b.value, 0, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to copy array to buffer\n");
        return ret;
    }

    printf("before building\n");
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to create program with source\n");
        return ret;
    }

    // Build the program
    ret = clBuildProgram(program, 1, &chosen_device, NULL, NULL, NULL);
    printf("ret at %d is %d\n", __LINE__, ret);

    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }
    else {
        printf("no errors\n");
    }

    printf("after building\n");
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &ret);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to create kernel\n");
        return ret;
    }

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to set kernel argument\n");
        return ret;
    }

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to set kernel argument\n");
        return ret;
    }

    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to set kernel argument\n");
        return ret;
    }

    ret = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&N);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to set kernel argument\n");
        return ret;
    }

    ret = clSetKernelArg(kernel, 4, sizeof(unsigned int), (void*)&K);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to set kernel argument\n");
        return ret;
    }

    ret = clSetKernelArg(kernel, 5, sizeof(unsigned int), (void*)&M);
    printf("ret at %d is %d\n", __LINE__, ret);
    if (ret) {
        printf("Failed to set kernel argument\n");
        return ret;
    }

    printf("before execution\n");
    // Count max work item dimensions
    size_t W = TILE_W;
    size_t H = TILE_H;

    // Execute the OpenCL kernel on the list
    size_t global_item_size[2] = { N , M };
    size_t local_item_size[2] = { W , H };
    cl_event event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
        global_item_size, local_item_size, 0, NULL, &event);
    if (ret) {
        printf("Execution failed\n");
        return ret;
    }
    printf("after execution\n");
    // Measure kernel time
    clWaitForEvents(1, &event);
    clFinish(command_queue);
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    cl_ulong time = double(time_end - time_start);
    printf("Total execution time: %0.5f milliseconds \n", time / 1000000.0);
    // Measure GFLOPS

    double GFLOPS = double(N) * double(M) * double(K) * 2.0 / time;
    printf("Total GFLOPS: %0.5lf GFLOPS\n", GFLOPS);
    // Read the memory buffer C on the device to the local variable C
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
        sizeof(float) * N * M, c.value, 0, NULL, NULL);
    printf("after copying\n");
    // Display the result to the screen
    bool is_successful = true;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            //printf("c[%zu][%zu] = %f || %f || %d\n", i, j, c[i * M + j], ac[i * M + j], (c[i * M + j] == ac[i * M + j]));
            if (c[i * M + j] != ac[i * M + j]) {
                is_successful = false;
            }
        }
    }
    if (is_successful) {
        printf("DONE CORRECTLY\n");
    }
    else {
        printf("FAILED\n");
    }
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(source_str);
    return 0;
}