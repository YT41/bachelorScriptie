#pragma once

#include "MemArena.hpp"
#include <cstddef>


typedef struct IntMatrix
{
    int32_t* data; /*ordered in column-major order, must have (rows * columns * elementByteSize) bytes available*/
    size_t rowCount;
    size_t columnCount;
} IntMatrix;

typedef struct Matrix
{
    double* data; /*ordered in column-major order, must have (rows * columns * elementByteSize) bytes available*/
    size_t rowCount;
    size_t columnCount;
} Matrix;

typedef struct Tensor
{
    double* data;
    size_t* dimensions;
    size_t dimensionCount;
} Tensor;


/*======================== general matrix functions ========================*/

static inline uint32_t GetIndex(uint32_t row, uint32_t column, uint32_t rowCount) { return (column * rowCount) + row; }


/*======================== Int matrix functions ========================*/

static inline size_t GetIntMatrixAllocSize(size_t rows, size_t columns) { return (rows * columns * sizeof(int32_t)); }
IntMatrix CreateBlankIntMatrix(MemArena* arena, size_t rows, size_t columns);

void IntMatrixAdd(IntMatrix* dest, IntMatrix A, IntMatrix B); /*dest = A + B*/
static inline void IntMatrixAddSelf(IntMatrix A, IntMatrix B) { IntMatrixAdd(&A, A, B); }; /*A = A + B*/
void IntMatrixAddValue(IntMatrix matrix, int32_t val, size_t row, size_t column); /*A_ij += val*/

int32_t GetValueIntMatrix(IntMatrix matrix, size_t row, size_t column);

void GetRowIntMatrix(IntMatrix matrix, int32_t* dest, size_t row);
void GetColumnDataIntMatrix(IntMatrix matrix, int32_t* dest, size_t column);
void GetColumnVectorIntMatrix(IntMatrix matrix, IntMatrix dest, size_t column);

void SetRowIntMatrix(IntMatrix matrix, const int32_t* src, size_t row);
void SetColumnIntMatrix(IntMatrix matrix, const int32_t* src, size_t column);

void SetValueIntMatrix(IntMatrix matrix, int32_t val, size_t row, size_t column);

void SetIntMatrix(IntMatrix matrix, int32_t val);

bool IntMatrixIsZero(IntMatrix matrix);


/*======================== matrix functions ========================*/

typedef double RealFn(double x);

static inline size_t GetMatrixAllocSize(size_t rows, size_t columns) { return (rows * columns * sizeof(double)); }
static inline size_t GetMatrixAllocSizeMatrix(Matrix A) { return (A.rowCount * A.columnCount * sizeof(double)); }
Matrix CreateMatrix(MemArena* arena, size_t rows, size_t columns, const double* vals);
Matrix CreateDiagonalMatrix(MemArena* arena, size_t rows, const double* diagVals);
Matrix CreateRandomMatrix(MemArena* arena, size_t rows, size_t columns);

Matrix CreateMatrixMultiply(MemArena* arena, Matrix A, Matrix B);
Matrix CreateMatrixScale(MemArena* arena, Matrix A, double lambda);

void MatrixMultiply(Matrix* dest, Matrix A, Matrix B); /*dest = AB*/
void MatrixAffineTransform(Matrix* dest, Matrix A, Matrix B, Matrix C); /*dest = AB + C*/
void MatrixAdd(Matrix* dest, Matrix A, Matrix B); /*dest = A + B*/
void MatrixSub(Matrix* dest, Matrix A, Matrix B); /*dest = A - B*/
void MatrixScale(Matrix* dest, Matrix A, double lambda); /*dest = lambda A*/
void MatrixTransform(Matrix* dest, Matrix A, RealFn sigma); /*dest_ij = sigma(A_ij)*/
void MatrixTransformDiagonal(Matrix* dest, Matrix A, RealFn sigma); /*dest_ii = sigma(A_ii)*/
void MatrixHadamard(Matrix* dest, Matrix A, Matrix B); /*dest_ij = A_ij * B_ij*/
double Dot(Matrix v, Matrix w); /*<v, w>, v and w must have 1 column*/

void MatrixAddValue(Matrix A, double val, size_t row, size_t column); /*A_ij += val*/
static inline void MatrixAddSelf(Matrix A, Matrix B) { MatrixAdd(&A, A, B); } /*A += B*/
static inline void MatrixSubSelf(Matrix A, Matrix B) { MatrixSub(&A, A, B); } /*A -= B*/
static inline void MatrixScaleSelf(Matrix A, double lambda) { MatrixScale(&A, A, lambda); } /*A = lambda A*/
static inline void MatrixTransformSelf(Matrix A, RealFn sigma) { MatrixTransform(&A, A, sigma); } /*A_ij = sigma(A_ij)*/
static inline void MatrixTransformDiagonalSelf(Matrix A, RealFn sigma) { MatrixTransformDiagonal(&A, A, sigma); } /*A_ii = sigma(A_ii)*/
static inline void MatrixHadamardSelf(Matrix A, Matrix B) { MatrixHadamard(&A, A, B); } /*A_ij = A_ij * B_ij*/

static inline double GetValueMatrix(Matrix matrix, size_t row, size_t column) { return (matrix.data[GetIndex(row, column, matrix.rowCount)]); }
double GetValueMatrixSafe(Matrix matrix, size_t row, size_t column);

void GetRowMatrix(Matrix matrix, double* dest, size_t row);
void GetColumnMatrix(Matrix matrix, double* dest, size_t column);

void SetRowMatrix(Matrix matrix, const double* src, size_t row);
void SetColumnMatrix(Matrix matrix, const double* src, size_t column);

void SetValueMatrix(Matrix matrix, double val, size_t row, size_t column);

void SetMatrix(Matrix matrix, double val);
void SetMatrixData(Matrix matrix, const double* vals);
void CopyMatrixData(Matrix dest, Matrix src);
void SetMatrixIdentity(Matrix A); /*A = I*/
void SetMatrixDiagonal(Matrix A, const double* diagVals);


/*======================== Tensor functions ========================*/

size_t GetTensorSize(size_t* dimensions, size_t dimensionCount);
size_t GetTensorAllocSize(size_t* dimensions, size_t dimensionCount);
Tensor CreateTensor(MemArena* arena, size_t* dimensions, size_t dimensionCount, const double* vals);

double GetValueTensor(Tensor tensor, IntMatrix indices);
void SetValueTensor(Tensor tensor, double val, IntMatrix indices);

void TensorScaleSelf(Tensor T, double lambda);
void TensorAddValue(Tensor T, double val, IntMatrix indices);

void SetTensor(Tensor tensor, double val);
void SetTensorData(Tensor tensor, const double* vals);


/*=============== print functions (for easy debugging) ===============*/

void PrintIntMatrix(IntMatrix matrix);
void PrintMatrix(Matrix matrix);