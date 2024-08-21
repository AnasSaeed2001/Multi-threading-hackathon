#include <vector>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <iomanip>
#include <mutex>

#include "MatrixOperations.h" 
#include "ThreadPool.hpp"
#include "Barrier.h"
#include "FileWrite.h"

// I defined a mutex here for locking to be used in certain sections later on in the code
std::mutex mtx;

void matrixOperationsInit(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    //This gets the size of the souurce matrix
    int dim = srcMatrix->size();

    // This section defines the matrices for operations 1 2 and 3
    std::vector<std::vector<double>> op1Matrix(dim);
    std::vector<std::vector<double>> op2Matrix(dim);
    std::vector<std::vector<double>> op3Matrix(dim);

    //This gets the number of available CPU cores
    int cpuCoreCount = std::thread::hardware_concurrency();

    //This changes the size of the destination matrix
    dstMatrix->resize(dim);

    //This for loop will resize and allocate memory for each operation matrix
    for (int i = 0; i < dim; i++)
    {
        op1Matrix[i].resize(dim);
        op2Matrix[i].resize(dim);
        op3Matrix[i].resize(dim);
        dstMatrix->at(i).resize(dim);
    }

    //This sections helps perform the operations in a sequence
    operation1(srcMatrix, &op1Matrix);
    operation2(&op1Matrix, &op2Matrix);
    operation3(&op2Matrix, &op3Matrix);

    //This for loop helps to copy the result to the destination matrix
    for (int i = 0; i < dim; i++)
    {
        dstMatrix->at(i).resize(dim);
        for (int j = 0; j < dim; j++)
        {
            dstMatrix->at(i).at(j) = op3Matrix.at(i).at(j);
        }
    }

    //This writes the matricies in a file
    //fileWrite("srcMatrix.txt", srcMatrix);
    //fileWrite("op1Matrix.txt", &op1Matrix);
    //fileWrite("op2Matrix.txt", &op2Matrix);
    //fileWrite("op3Matrix.txt", &op3Matrix);
    //fileWrite("dstMatrix.txt", dstMatrix);
}



//Operation 1 - Matrix Transposition
void operation1(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    //Gets the number of available CPU cores
    int numThreads = std::thread::hardware_concurrency();
    //This gets the number of rows in the source matrix
    int numRows = srcMatrix->size();
    int chunkSize = numRows / numThreads;

    //This will create a thread pool with the number of hardware threads
    ThreadPool pool(numThreads);

    //This loop queues tasks for each element in the matrix
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numRows; ++j) {
            pool.enqueue([i, j, srcMatrix, dstMatrix]() {
                //This locks the mutex
                std::lock_guard<std::mutex> lock(mtx); 
                //This performs the matrix transpositions
                dstMatrix->at(j).at(i) = srcMatrix->at(i).at(j);
                });
        }
    }
}
;

//Operation 2 - Zone Sum
void operation2(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{   
    //This gets the number of rows in the source matrix
    int numRows = srcMatrix->size();
    //This gets the number of columns in the source matrix
    int numCols = srcMatrix->at(0).size();
    //Number of cpu cores again
    int numThreads = std::thread::hardware_concurrency();

    //Calculates the number of elements each thread will process
    int elementsPerThread = (numRows * numCols) / numThreads;

    //Vector to store the threads
    std::vector<std::thread> threads(numThreads);

    //This defines a function to perform the operation for a range of elements
    auto operationFunction = [&](int start, int end) {
        for (int index = start; index < end; ++index) {
            int i = index / numCols;
            int j = index % numCols;

            //This performs the operation on the current element (i, j)
            double sum = 0;
            int count = 0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    int x = i + dx;
                    int y = j + dy;
                    if (x >= 0 && x < numRows && y >= 0 && y < numCols) {
                        sum += srcMatrix->at(x).at(y);
                        count++;
                    }
                }
            }
            dstMatrix->at(i).at(j) = sum;
        }
        };

    //This launches all of the threads
    int start = 0;
    for (int t = 0; t < numThreads; ++t) {
        int end = (t == numThreads - 1) ? numRows * numCols : start + elementsPerThread;
        threads[t] = std::thread(operationFunction, start, end);
        start = end;
    }

    //This joins the threads
    for (auto& thread : threads) {
        thread.join();
    }

}

//Operation 3 - Matrix Multiplication
void operation3(std::vector<std::vector<double>> * srcMatrix, std::vector<std::vector<double>> * dstMatrix)
{
    //This gets the number of rows in the source matrix
    int numRows = srcMatrix->size();
    //This get the number of columns in the source matrix
    int numCols = srcMatrix->at(0).size();
    //CPU cores again
    int numThreads = std::thread::hardware_concurrency();

    //This creates a thread pool with the number of hardware threads
    ThreadPool pool(numThreads); // Create a thread pool with the number of hardware threads

    //Queues tasks for each element in the matrix
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            pool.enqueue([i, j, srcMatrix, dstMatrix, numRows]() {
                double sum = 0;
                for (int k = 0; k < numRows; ++k) {
                    sum += srcMatrix->at(i).at(k) * srcMatrix->at(k).at(j);
                }
                std::lock_guard<std::mutex> lock(mtx);
                //This performs the matrix multiplication
                dstMatrix->at(i).at(j) = sum;
                });
        }
    }
}
