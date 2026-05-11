#include "Matrix.hpp"
#include "MemArena.hpp"
#include "Random.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>


static inline bool RowWithinIntMatrixBounds(IntMatrix matrix, size_t row)
{
    return (row < (matrix.rowCount));
}

static inline bool ColumnWithinIntMatrixBounds(IntMatrix matrix, size_t column)
{
    return (column < (matrix.columnCount));
}


IntMatrix CreateBlankIntMatrix(MemArena* arena, size_t rows, size_t columns)
{
    return  (IntMatrix){ (int32_t*)MemArenaAlloc(arena, GetIntMatrixAllocSize(rows, columns)), rows, columns };
}

void IntMatrixAdd(IntMatrix* dest, IntMatrix A, IntMatrix B)
{
    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < A.columnCount; j++)
            dest->data[GetIndex(i, j, (dest->rowCount))] = (A.data[GetIndex(i, j, A.rowCount)] + B.data[GetIndex(i, j, B.rowCount)]);
    }   
}

void IntMatrixAddValue(IntMatrix matrix, int32_t val, size_t row, size_t column)
{
    SetValueIntMatrix(matrix, (GetValueIntMatrix(matrix, row, column) + val), row, column);
}

int32_t GetValueIntMatrix(IntMatrix matrix, size_t row, size_t column)
{
    if(RowWithinIntMatrixBounds(matrix, row) && ColumnWithinIntMatrixBounds(matrix, column))
        return (matrix.data[GetIndex(row, column, matrix.rowCount)]);
    return INT32_MAX;
}

void GetRowIntMatrix(IntMatrix matrix, int32_t* dest, size_t row)
{
    if(RowWithinIntMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.columnCount; i++)
            dest[i] = matrix.data[GetIndex(row, i, matrix.rowCount)];
    }
}

void GetColumnDataIntMatrix(IntMatrix matrix, int32_t* dest, size_t column)
{
    if(ColumnWithinIntMatrixBounds(matrix, column))
        memmove(dest, (matrix.data + (column * matrix.rowCount)), (matrix.rowCount * sizeof(int32_t)));
}

void GetColumnVectorIntMatrix(IntMatrix matrix, IntMatrix dest, size_t column)
{
    if(ColumnWithinIntMatrixBounds(matrix, column))
        memmove((dest.data), (matrix.data + (column * matrix.rowCount)), (matrix.rowCount * sizeof(int32_t)));
}

void SetRowIntMatrix(IntMatrix matrix, const int32_t* src, size_t row)
{
    if(RowWithinIntMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.columnCount; i++)
            matrix.data[GetIndex(row, i, matrix.rowCount)] = src[i];
    }
}

void SetColumnIntMatrix(IntMatrix matrix, const int32_t* src, size_t column)
{
    if(ColumnWithinIntMatrixBounds(matrix, column))
        memmove((matrix.data + (column * matrix.rowCount)), src, (matrix.rowCount * sizeof(int32_t)));
}

void SetValueIntMatrix(IntMatrix matrix, int32_t val, size_t row, size_t column)
{
    if(RowWithinIntMatrixBounds(matrix, row) && ColumnWithinIntMatrixBounds(matrix, column))
        matrix.data[GetIndex(row, column, matrix.rowCount)] = val;
}

void SetIntMatrix(IntMatrix matrix, int32_t val)
{
    for(uint32_t x = 0; x < matrix.rowCount; x++)
    {
        for(uint32_t y = 0; y < matrix.columnCount; y++)
            SetValueIntMatrix(matrix, val, x, y);
    }
}

bool IntMatrixIsZero(IntMatrix matrix)
{
    for(size_t x = 0; x < matrix.rowCount; x++)
    {
        for(size_t y = 0; y < matrix.columnCount; y++)
        {
            if(GetValueIntMatrix(matrix, x, y) != 0)
                return false;
        }
    }
    return true;
}


/*======================== matrix functions ========================*/

static inline bool RowWithinMatrixBounds(Matrix matrix, size_t row)
{
    return (row < (matrix.rowCount));
}

static inline bool ColumnWithinMatrixBounds(Matrix matrix, size_t column)
{
    return (column < (matrix.columnCount));
}

static inline bool IsSquareMatrix(Matrix A)
{
    return ((A.rowCount) == (A.columnCount));
}


Matrix CreateMatrix(MemArena* arena, size_t rows, size_t columns, const double* vals)
{
    Matrix ret = { (double*)MemArenaAlloc(arena, GetMatrixAllocSize(rows, columns)), rows, columns };

    if(vals != NULL)
        SetMatrixData(ret, vals);

    return ret;
}

Matrix CreateDiagonalMatrix(MemArena* arena, size_t rows, const double* diagVals)
{
    Matrix ret = CreateMatrix(arena, rows, rows, NULL);
    SetMatrixDiagonal(ret, diagVals);

    return ret;
}

Matrix CreateRandomMatrix(MemArena* arena, size_t rows, size_t columns)
{
    Matrix ret = CreateMatrix(arena, rows, columns, NULL);

    for(uint32_t i = 0; i < rows; i++)
    {
        for(uint32_t j = 0; j < columns; j++)
            SetValueMatrix(ret, StandardOpenUniformSim(), i, j);
    }

    return ret;
}

Matrix CreateMatrixMultiply(MemArena* arena, Matrix A, Matrix B)
{
    Matrix ret = CreateMatrix(arena, A.rowCount, B.columnCount, NULL);
    MatrixMultiply(&ret, A, B);

    return ret;
}

Matrix CreateMatrixScale(MemArena* arena, Matrix A, double lambda)
{
    Matrix ret = CreateMatrix(arena, A.rowCount, A.columnCount, NULL);
    MatrixScale(&ret, A, lambda);

    return ret;
}

void MatrixMultiply(Matrix* dest, Matrix A, Matrix B)
{
    if(A.columnCount != B.rowCount) 
        return;

    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < B.columnCount; j++)
        {
            double val = 0.0;
            for(uint32_t k = 0; k < A.columnCount; k++)
                val += (GetValueMatrix(A, i, k) * GetValueMatrix(B, k, j));
            dest->data[GetIndex(i, j, (dest->rowCount))] = val;
        }
    }
}

void MatrixAffineTransform(Matrix* dest, Matrix A, Matrix B, Matrix C)
{
    if(A.columnCount != B.rowCount) 
        return;

    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < B.columnCount; j++)
        {
            double val = 0.0;
            for(uint32_t k = 0; k < A.columnCount; k++)
                val += (GetValueMatrix(A, i, k) * GetValueMatrix(B, k, j));
            dest->data[GetIndex(i, j, (dest->rowCount))] = val + GetValueMatrix(C, i, j);
        }
    }
}

void MatrixAddValue(Matrix A, double val, size_t row, size_t column)
{
    SetValueMatrix(A, (GetValueMatrix(A, row, column) + val), row, column);
}

void MatrixAdd(Matrix* dest, Matrix A, Matrix B)
{
    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < A.columnCount; j++)
            dest->data[GetIndex(i, j, (dest->rowCount))] = (A.data[GetIndex(i, j, A.rowCount)] + B.data[GetIndex(i, j, B.rowCount)]);
    }
}

void MatrixSub(Matrix* dest, Matrix A, Matrix B)
{
    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < A.columnCount; j++)
            dest->data[GetIndex(i, j, (dest->rowCount))] = (A.data[GetIndex(i, j, A.rowCount)] - B.data[GetIndex(i, j, B.rowCount)]);
    }
}

void MatrixScale(Matrix* dest, Matrix A, double lambda)
{
    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < A.columnCount; j++)
            dest->data[GetIndex(i, j, (dest->rowCount))] = lambda * A.data[GetIndex(i, j, A.rowCount)];
    }
}

void MatrixTransform(Matrix* dest, Matrix A, RealFn sigma)
{
    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < A.columnCount; j++)
            dest->data[GetIndex(i, j, (dest->rowCount))] = sigma(A.data[GetIndex(i, j, A.rowCount)]);
    }
}

void MatrixTransformDiagonal(Matrix* dest, Matrix A, RealFn sigma)
{
    if(IsSquareMatrix(A))
    {
        for(uint32_t i = 0; i < A.rowCount; i++)
            SetValueMatrix(*dest, sigma(GetValueMatrix(A, i, i)), i, i);
    }
}

void MatrixHadamard(Matrix* dest, Matrix A, Matrix B)
{
    for(uint32_t i = 0; i < A.rowCount; i++)
    {
        for(uint32_t j = 0; j < A.columnCount; j++)
            dest->data[GetIndex(i, j, (dest->rowCount))] = (A.data[GetIndex(i, j, A.rowCount)] * B.data[GetIndex(i, j, B.rowCount)]);
    }
}

double Dot(Matrix v, Matrix w)
{
    uint32_t d = v.rowCount;
    double sum = 0.0;
    for(uint32_t i = 0; i < d; i++)
        sum += (v.data[GetIndex(i, 0, d)] * w.data[GetIndex(i, 0, d)]);
    return sum;
}

double GetValueMatrixSafe(Matrix matrix, size_t row, size_t column)
{
    if(RowWithinMatrixBounds(matrix, row) && ColumnWithinMatrixBounds(matrix, column))
        return GetValueMatrix(matrix, row, column);
    return NAN;
}

void GetRowMatrix(Matrix matrix, double* dest, size_t row)
{
    if(RowWithinMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.columnCount; i++)
            dest[i] = matrix.data[GetIndex(row, i, matrix.rowCount)];
    }
}
void GetColumnMatrix(Matrix matrix, double* dest, size_t column)
{
    if(ColumnWithinMatrixBounds(matrix, column))
        memmove(dest, (matrix.data + (column * matrix.rowCount)), (matrix.rowCount * sizeof(double)));
}

void SetRowMatrix(Matrix matrix, const double* src, size_t row)
{
    if(RowWithinMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.columnCount; i++)
            matrix.data[GetIndex(row, i, matrix.rowCount)] = src[i];
    }
}
void SetColumnMatrix(Matrix matrix, const double* src, size_t column)
{
    if(ColumnWithinMatrixBounds(matrix, column))
        memmove((matrix.data + (column * matrix.rowCount)), src, (matrix.rowCount * sizeof(double)));
}

void SetValueMatrix(Matrix matrix, double val, size_t row, size_t column)
{
    if(RowWithinMatrixBounds(matrix, row) && ColumnWithinMatrixBounds(matrix, column))
        matrix.data[GetIndex(row, column, matrix.rowCount)] = val;
}

void SetMatrix(Matrix matrix, double val)
{
    for(uint32_t x = 0; x < matrix.rowCount; x++)
    {
        for(uint32_t y = 0; y < matrix.columnCount; y++)
            SetValueMatrix(matrix, val, x, y);
    }
}

void SetMatrixData(Matrix matrix, const double* vals)
{
    memmove((void*)(matrix.data), (void*)vals, GetMatrixAllocSize(matrix.rowCount, matrix.columnCount));
}

void CopyMatrixData(Matrix dest, Matrix src)
{
    memmove((void*)(dest.data), (void*)(src.data), GetMatrixAllocSize(src.rowCount, src.columnCount));
}

void SetMatrixIdentity(Matrix A)
{
    if(IsSquareMatrix(A))
    {
        SetMatrix(A, 0.0);
        for(uint32_t i = 0; i < (A.rowCount); i++)
            SetValueMatrix(A, 1.0, i, i);
    }
}

void SetMatrixDiagonal(Matrix A, const double* diagVals)
{
    if(IsSquareMatrix(A))
    {
        SetMatrix(A, 0.0);
        for(uint32_t i = 0; i < (A.rowCount); i++)
            SetValueMatrix(A, diagVals[i], i, i);
    }
}


/*======================== Tensor functions ========================*/

static inline size_t GetDataIndexTensor(Tensor tensor, IntMatrix indices)
{ 
    size_t sum = 0;
    size_t product = 1;
    for(size_t i = 0; i < tensor.dimensionCount; i++)
    {
        sum += (product * GetValueIntMatrix(indices, i, 0));
        product *= tensor.dimensions[i];
    }
    return sum; 
}

static inline size_t GetTensorDataAllocSize(size_t* dimensions, size_t dimensionCount)
{
    return (GetTensorSize(dimensions, dimensionCount) * sizeof(double));
}

size_t GetTensorSize(size_t* dimensions, size_t dimensionCount)
{
    size_t product = 1;
    for(size_t i = 0; i < dimensionCount; i++)
        product *= dimensions[i];
    return product;
}

size_t GetTensorAllocSize(size_t* dimensions, size_t dimensionCount)
{
    return GetTensorDataAllocSize(dimensions, dimensionCount) + (dimensionCount * sizeof(size_t));
}

Tensor CreateTensor(MemArena* arena, size_t* dimensions, size_t dimensionCount, const double* vals)
{
    Tensor ret = { 
        (double*)MemArenaAlloc(arena, GetTensorDataAllocSize(dimensions, dimensionCount)), 
        (size_t*)MemArenaAlloc(arena, (dimensionCount * sizeof(size_t))),
        dimensionCount
    };
    memcpy(ret.dimensions, dimensions, (dimensionCount * sizeof(size_t)));

    if(vals != NULL)
        SetTensorData(ret, vals);

    return ret;
}

double GetValueTensor(Tensor tensor, IntMatrix indices)
{
    return tensor.data[GetDataIndexTensor(tensor, indices)];
}

void SetValueTensor(Tensor tensor, double val, IntMatrix indices)
{
    tensor.data[GetDataIndexTensor(tensor, indices)] = val;
}

void TensorScaleSelf(Tensor T, double lambda)
{
    for(size_t i = 0; i < GetTensorSize(T.dimensions, T.dimensionCount); i++)
        T.data[i] *= lambda;
}

void TensorAddValue(Tensor T, double val, IntMatrix indices)
{
    SetValueTensor(T, (GetValueTensor(T, indices) + val), indices);
}

void SetTensor(Tensor tensor, double val)
{
    for(size_t i = 0; i < GetTensorSize(tensor.dimensions, tensor.dimensionCount); i++)
        tensor.data[i] = val;
}

void SetTensorData(Tensor tensor, const double* vals)
{
    for(size_t i = 0; i < GetTensorSize(tensor.dimensions, tensor.dimensionCount); i++)
        tensor.data[i] = vals[i];
}


/*=============== print functions (for easy debugging) ===============*/

void PrintIntMatrix(IntMatrix matrix)
{
    for(uint32_t r = 0; r < matrix.rowCount; r++)
    {
        printf("( ");
        for(uint32_t c = 0; c < matrix.columnCount; c++)
            printf("%3i ", GetValueIntMatrix(matrix, r, c));
        printf(")\n");
    }
    printf("\n");
}

void PrintMatrix(Matrix matrix)
{
    for(uint32_t r = 0; r < matrix.rowCount; r++)
    {
        printf("( ");
        for(uint32_t c = 0; c < matrix.columnCount; c++)
            printf("%5.3f ", GetValueMatrixSafe(matrix, r, c));
        printf(")\n");
    }
    printf("\n");
}