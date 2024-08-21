#pragma once

#include <vector>
#include "Barrier.h"
#include <stdexcept>
#include <thread>
#include <iostream>
#include <iomanip>
#include <mutex>

#include "MatrixOperations.h" 
#include "ThreadPool.hpp"
#include "FileWrite.h"

void matrixOperationsInit(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix);
void operation1(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix);
void operation2(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix);
void operation3(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix);
