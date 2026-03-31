#include "Matrix.hpp"
#include "Random.hpp"

#include <cmath>
#include <cstdint>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>


static inline bool RowWithinIntMatrixBounds(IntMatrixNxM matrix, size_t row)
{
    return (row < (matrix.rowCount));
}

static inline bool ColumnWithinIntMatrixBounds(IntMatrixNxM matrix, size_t column)
{
    return (column < (matrix.columnCount));
}


IntMatrixNxM CreateBlankIntMatrix(MemArena* arena, size_t rows, size_t columns)
{
    return  (IntMatrixNxM){ (int32_t*)MemArenaAlloc(arena, GetIntMatrixAllocSize(rows, columns)), rows, columns };
}

int32_t GetValueIntMatrix(IntMatrixNxM matrix, size_t row, size_t column)
{
    if(RowWithinIntMatrixBounds(matrix, row) && ColumnWithinIntMatrixBounds(matrix, column))
        return (matrix.matrixData[GetIndex(row, column, matrix.rowCount)]);
    return INT32_MAX;
}

void GetRowIntMatrix(IntMatrixNxM matrix, int32_t* dest, size_t row)
{
    if(RowWithinIntMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.columnCount; i++)
            dest[i] = matrix.matrixData[GetIndex(row, i, matrix.rowCount)];
    }
}

void GetColumnIntMatrix(IntMatrixNxM matrix, int32_t* dest, size_t column)
{
    if(ColumnWithinIntMatrixBounds(matrix, column))
        memmove(dest, (matrix.matrixData + (column * matrix.rowCount)), (matrix.rowCount * sizeof(int32_t)));
}

void SetRowIntMatrix(IntMatrixNxM matrix, const int32_t* src, size_t row)
{
    if(RowWithinIntMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.columnCount; i++)
            matrix.matrixData[GetIndex(row, i, matrix.rowCount)] = src[i];
    }
}

void SetColumnIntMatrix(IntMatrixNxM matrix, const int32_t* src, size_t column)
{
    if(ColumnWithinIntMatrixBounds(matrix, column))
        memmove((matrix.matrixData + (column * matrix.rowCount)), src, (matrix.rowCount * sizeof(int32_t)));
}

void SetValueIntMatrix(IntMatrixNxM matrix, int32_t val, size_t row, size_t column)
{
    if(RowWithinIntMatrixBounds(matrix, row) && ColumnWithinIntMatrixBounds(matrix, column))
        matrix.matrixData[GetIndex(row, column, matrix.rowCount)] = val;
}

void SetIntMatrix(IntMatrixNxM matrix, int32_t val)
{
    for(uint32_t x = 0; x < matrix.rowCount; x++)
    {
        for(uint32_t y = 0; y < matrix.columnCount; y++)
            SetValueIntMatrix(matrix, val, x, y);
    }
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


Matrix CreateMatrix(MemArena* arena, size_t rows, size_t columns, double* vals)
{
    Matrix ret = { (double*)MemArenaAlloc(arena, GetMatrixAllocSize(rows, columns)), rows, columns };

    if(vals != NULL)
        SetMatrixData(ret, vals);

    return ret;
}

Matrix CreateDiagonalMatrix(MemArena* arena, size_t rows, double* diagVals)
{
    Matrix ret = CreateMatrix(arena, rows, rows, NULL);

    SetMatrix(ret, 0.0);
    if(diagVals != NULL)
    {
        for(uint32_t i = 0; i < rows; i++)
            SetValueMatrix(ret, diagVals[i], i, i);
    }
    else 
    {
        for(uint32_t i = 0; i < rows; i++)
            SetValueMatrix(ret, 1.0, i, i);
    }

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
                val += (A.data[GetIndex(i, k, A.rowCount)] * B.data[GetIndex(k, j, B.rowCount)]);
            dest->data[GetIndex(i, j, (dest->rowCount))] = val;
        }
    }
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

void SetMatrixData(Matrix matrix, double* vals)
{
    memmove((void*)(matrix.data), (void*)vals, GetMatrixAllocSize(matrix.rowCount, matrix.columnCount));
}

void CopyMatrixData(Matrix dest, Matrix src)
{
    memmove((void*)(dest.data), (void*)(src.data), GetMatrixAllocSize(src.rowCount, src.columnCount));
}


/*=============== print functions (for easy debugging) ===============*/

void PrintIntMatrix(IntMatrixNxM matrix)
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
            printf("%5.1f ", GetValueMatrixSafe(matrix, r, c));
        printf(")\n");
    }
    printf("\n");
}