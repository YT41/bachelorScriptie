#pragma once

#include "MemArena.hpp"


/*abstract datatype for nxm matrices of any element type*/
typedef struct IntMatrixNxM
{
    int32_t* matrixData; /*ordered in column-major order, must have (rows * columns * elementByteSize) bytes available*/
    size_t RowCount;
    size_t ColumnCount;
} IntMatrixNxM;


static inline size_t GetIntMatrixAllocSize(size_t rows, size_t columns) { return (rows * columns * sizeof(int32_t)); }
IntMatrixNxM CreateBlankIntMatrix(MemArena* arena, size_t rows, size_t columns);

int32_t GetValueMatrix(IntMatrixNxM matrix, size_t row, size_t column);

void GetRowMatrix(IntMatrixNxM matrix, int32_t* dest, size_t row);
void GetColumnMatrix(IntMatrixNxM matrix, int32_t* dest, size_t column);

void SetRowMatrix(IntMatrixNxM matrix, const int32_t* src, size_t row);
void SetColumnMatrix(IntMatrixNxM matrix, const int32_t* src, size_t column);

void SetValueMatrix(IntMatrixNxM matrix, int32_t val, size_t row, size_t column);

void SetMatrix(IntMatrixNxM, int32_t val);


/*=============== print functions (for easy debugging) ===============*/

void PrintMatrix(IntMatrixNxM matrix);