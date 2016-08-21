#include "Gaussian_Blur.h"

Mat gaussian_blur(Mat image)
{
	/* Host/device data structures */
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_int err;

	/* Program/kernel data structures */
	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	cl_kernel kernel;

	/*buffers */
	cl_mem src_R_buff, src_G_buff, src_B_buff;
	cl_mem dst_R_buff, dst_G_buff, dst_B_buff;
	cl_mem Filter_buff;
	cl_mem param_buff;
	size_t work_units_per_kernel;

	int param[3];
	param[0] = image.cols;
	param[1] = image.rows;

	unsigned char *src_R = new unsigned char[image.cols*image.rows];
	unsigned char *src_G = new unsigned char[image.cols*image.rows];
	unsigned char *src_B = new unsigned char[image.cols*image.rows];

	unsigned char *dst_R = new unsigned char[image.cols*image.rows];
	unsigned char *dst_G = new unsigned char[image.cols*image.rows];
	unsigned char *dst_B = new unsigned char[image.cols*image.rows];


	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
		{
			src_B[i*image.cols + j] = image.at<Vec3b>(i, j)[0];
			src_G[i*image.cols + j] = image.at<Vec3b>(i, j)[1];
			src_R[i*image.cols + j] = image.at<Vec3b>(i, j)[2];
		}
	int FilterSize;
	printf("GaussianSize : ");
	std::cin >> FilterSize;
	param[2] = FilterSize;
	float *GaussianFilter = new float[FilterSize*FilterSize];
	for (int i = 0; i < FilterSize; i++)
		for (int j = 0; j < FilterSize; j++)
			GaussianFilter[i*FilterSize + j] = exp(-1 * ((i - (FilterSize >> 1))*(i - (FilterSize >> 1)) + (j - (FilterSize >> 1))*(j - (FilterSize >> 1))) / (2 * sigma*sigma)) / (2 * pi*sigma*sigma);

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't find any platforms");
		exit(-1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1,
		&device, NULL);
	if (err < 0) {
		perror("Couldn't find any devices");
		exit(-1);
	}

	/* Create the context */
	context = clCreateContext(NULL, 1, &device, NULL,
		NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(-1);
	}

	/* Read program file and place content into buffer */
	program_handle = fopen(PROGRAM_FILE, "r");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(-1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(context, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(-1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(-1);
	}

	/* Create kernel for the KERNEL_FUNC function */
	kernel = clCreateKernel(program, KERNEL_FUNC, &err);
	if (err < 0) {
		perror("Couldn't create the kernel");
		exit(-1);
	}

	/* Create CL buffers to hold input and output data */
	src_R_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)* image.cols*image.rows, src_R, &err);
	src_G_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)* image.cols*image.rows, src_G, &err);
	src_B_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)* image.cols*image.rows, src_B, &err);
	Filter_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(float)* FilterSize*FilterSize, GaussianFilter, &err);
	param_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, sizeof(int) * 3, param, &err);
	if (err < 0) {
		perror("Couldn't create a buffer object");
		exit(-1);
	}
	dst_R_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(unsigned char)* image.cols*image.rows, NULL, NULL);
	dst_G_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(unsigned char)* image.cols*image.rows, NULL, NULL);
	dst_B_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(unsigned char)* image.cols*image.rows, NULL, NULL);

	/* Create kernel arguments from the CL buffers */
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_R_buff);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_G_buff);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &src_B_buff);
	if (err < 0) {
		perror("Couldn't set the kernel argument");
		exit(-1);
	}
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_R_buff);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &dst_G_buff);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &dst_B_buff);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &Filter_buff);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &param_buff);

	/* Create a CL command queue for the device*/
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err < 0) {
		perror("Couldn't create the command queue");
		exit(-1);
	}

	/* Enqueue the command queue to the device */
	work_units_per_kernel = image.cols*image.rows; /* 4 work-units per kernel */

	cl_event event;
	cl_ulong time_start;
	cl_ulong time_end;

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
		NULL, 0, NULL, &event);
	clWaitForEvents(1, &event);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	printf("GPU : %0.3f  ms\n", ((time_end - time_start) / 1000000.0));

	if (err < 0) {
		perror("Couldn't enqueue the kernel execution command");
		exit(-1);
	}

	/* Read the result */
	err = clEnqueueReadBuffer(queue, dst_R_buff, CL_TRUE, 0, sizeof(unsigned char)* image.cols*image.rows,
		dst_R, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue, dst_G_buff, CL_TRUE, 0, sizeof(unsigned char)* image.cols*image.rows,
		dst_G, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue, dst_B_buff, CL_TRUE, 0, sizeof(unsigned char)* image.cols*image.rows,
		dst_B, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the read buffer command");
		exit(-1);
	}

	unsigned char tmp_B = 0;
	unsigned char tmp_G = 0;
	unsigned char tmp_R = 0;

	clock_t start_time, end_time;
	float total_time = 0;
	start_time = clock();
	for (int i = (FilterSize >> 1); i < image.rows - (FilterSize >> 1); i++)
		for (int j = (FilterSize >> 1); j < image.cols - (FilterSize >> 1); j++)
			for (int m = 0; m < FilterSize; m++)
				for (int n = 0; n < FilterSize; n++)
				{
					tmp_B += src_B[(i - (FilterSize >> 1) + m)*image.cols + j - (FilterSize >> 1) + n] * GaussianFilter[m*FilterSize + n];
					tmp_G += src_G[(i - (FilterSize >> 1) + m)*image.cols + j - (FilterSize >> 1) + n] * GaussianFilter[m*FilterSize + n];
					tmp_R += src_R[(i - (FilterSize >> 1) + m)*image.cols + j - (FilterSize >> 1) + n] * GaussianFilter[m*FilterSize + n];
				}
	end_time = clock();
	total_time = (float)(end_time - start_time);
	printf("CPU : %0.3f ms \n", total_time);

	Mat new_image = Mat::zeros(image.rows, image.cols, image.type());
	
	for (int i = 0; i < new_image.rows; i++)
		for (int j = 0; j < new_image.cols; j++)
		{
			new_image.at<Vec3b>(i, j)[0] = (uchar)dst_B[i*image.cols + j];
			new_image.at<Vec3b>(i, j)[1] = (uchar)dst_G[i*image.cols + j];
			new_image.at<Vec3b>(i, j)[2] = (uchar)dst_R[i*image.cols + j];
		}

	delete[] src_R;
	delete[] src_G;
	delete[] src_B;
	delete[] dst_R;
	delete[] dst_G;
	delete[] dst_B;
	delete[] GaussianFilter;
	/*

	/// Create Windows
	namedWindow("Original Image", 1);
	namedWindow("New Image", 1);

	/// Show stuff
	imshow("Original Image", image);
	imshow("New Image", new_image);

	waitKey(1);*/

	/* Deallocate resources */
	clReleaseMemObject(src_R_buff);
	clReleaseMemObject(src_G_buff);
	clReleaseMemObject(src_B_buff);
	clReleaseMemObject(dst_R_buff);
	clReleaseMemObject(dst_G_buff);
	clReleaseMemObject(dst_B_buff);
	clReleaseMemObject(Filter_buff);
	clReleaseMemObject(param_buff);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	clReleaseEvent(event);

	image.release();
	return new_image;
};