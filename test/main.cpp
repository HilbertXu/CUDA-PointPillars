/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <fstream>

#include "cuda_runtime.h"

#include "./params.h"
#include "./pointpillar.h"
#include "./csvFile.h"

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

std::string Data_File = "../../seq/";
std::string Model_File = "../../model/pointpillar.onnx";
std::string Target_Dir = "/home/hilbertxu/puzek_ws/CUDA-PointPillars/resutls/";

void Getinfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[len];
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

int main(int argc, const char **argv)
{
  Getinfo();

  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaStream_t stream = NULL;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaStreamCreate(&stream));

  Params params_;

  std::vector<Bndbox> nms_pred;
  nms_pred.reserve(100);

  PointPillar pointpillar(Model_File, stream);

  for (int i = 0; i < 559; i++)
  {
    std::string dataFile = Data_File;

    std::stringstream ss;

    ss<< i;

    if (i<10) {
      dataFile +="00000";
      dataFile += ss.str();
      dataFile +=".bin";
    }
    else if(i< 100) {
      dataFile +="0000";
      dataFile += ss.str();
      dataFile +=".bin";
    }
    else {
      dataFile +="000";
      dataFile += ss.str();
      dataFile +=".bin";
    }
    

    std::string csvFileName = Target_Dir;
    csvFileName += "00000";
    csvFileName += ss.str();
    csvFileName += ".csv";
    std::string fileID = "00000" + ss.str();

    std::cout << "<<<<<<<<<<<" <<std::endl;
    std::cout << "load file: "<< dataFile <<std::endl;

    csvfile csv(csvFileName);

    csv << fileID << "x" << "y" << "z" << "w" << "l" << "h" << "rt" << "id" << "score" << endrow;

    //load points cloud
    unsigned int length = 0;
    void *data = NULL;
    std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
    loadData(dataFile.data(), &data, &length);
    buffer.reset((char *)data);

    float* points = (float*)buffer.get();
    size_t points_size = length/sizeof(float)/4;

    std::cout << "find points num: "<< points_size <<std::endl;

    float *points_data = nullptr;
    unsigned int points_data_size = points_size * 4 * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
    checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(start, stream);

    pointpillar.doinfer(points_data, points_size, nms_pred);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout<<"TIME: pointpillar: "<< elapsedTime <<" ms." <<std::endl;

    checkCudaErrors(cudaFree(points_data));

    std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;
    // nms_pred.clear();
    std::cout << ">>>>>>>>>>>" <<std::endl;
    for (Bndbox box: nms_pred){
      std::cout << "detected box: " << box.x << " " << box.y << " " << box.z << " "
                << box.w << " " << box.l << " " << box.h << " "
                << box.rt << " " << box.id << " " << box.score << " " << std::endl;
      
      csv << box.x << box.y << box.z << box.w << box.l << box.h << box.rt << box.id << box.score << endrow;
    }
    std::cout << "<<<<<<<<<<<" << std::endl;
    nms_pred.clear();
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaStreamDestroy(stream));

  return 0;
}
