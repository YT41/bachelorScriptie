#include "Matrix.hpp"

#include <string.h>
#include <stdbool.h>
#include <stdio.h>



static inline bool RowWithinMatrixBounds(IntMatrixNxM matrix, size_t row)
{
    return (row < (matrix.RowCount));
}

static inline bool ColumnWithinMatrixBounds(IntMatrixNxM matrix, size_t column)
{
    return (column < (matrix.ColumnCount));
}


IntMatrixNxM CreateBlankIntMatrix(MemArena* arena, size_t rows, size_t columns)
{
    return  (IntMatrixNxM){ (int32_t*)MemArenaAlloc(arena, GetIntMatrixAllocSize(rows, columns)), rows, columns };
}

int32_t GetValueMatrix(IntMatrixNxM matrix, size_t row, size_t column)
{
    if(RowWithinMatrixBounds(matrix, row) && ColumnWithinMatrixBounds(matrix, column))
        return (matrix.matrixData[(column * (matrix.RowCount)) + row]);
    return INT32_MAX;
}

void GetRowMatrix(IntMatrixNxM matrix, int32_t* dest, size_t row)
{
    if(RowWithinMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.ColumnCount; i++)
            dest[i] = matrix.matrixData[(i * (matrix.RowCount)) + row];
    }
}

void GetColumnMatrix(IntMatrixNxM matrix, int32_t* dest, size_t column)
{
    if(ColumnWithinMatrixBounds(matrix, column))
        memmove(dest, (matrix.matrixData + (column * matrix.RowCount)), (matrix.RowCount * sizeof(int32_t)));
}

void SetRowMatrix(IntMatrixNxM matrix, const int32_t* src, size_t row)
{
    if(RowWithinMatrixBounds(matrix, row))
    {
        for(uint32_t i = 0; i < matrix.ColumnCount; i++)
            matrix.matrixData[(i * (matrix.RowCount)) + row] = src[i];
    }
}

void SetColumnMatrix(IntMatrixNxM matrix, const int32_t* src, size_t column)
{
    if(ColumnWithinMatrixBounds(matrix, column))
        memmove((matrix.matrixData + (column * matrix.RowCount)), src, (matrix.RowCount * sizeof(int32_t)));
}

void SetValueMatrix(IntMatrixNxM matrix, int32_t val, size_t row, size_t column)
{
    if(RowWithinMatrixBounds(matrix, row) && ColumnWithinMatrixBounds(matrix, column))
        matrix.matrixData[(column * matrix.RowCount) + row] = val;
}

void SetMatrix(IntMatrixNxM matrix, int32_t val)
{
    for(uint32_t x = 0; x < matrix.RowCount; x++)
    {
        for(uint32_t y = 0; y < matrix.ColumnCount; y++)
            SetValueMatrix(matrix, val, x, y);
    }
}


/*=============== print functions (for easy debugging) ===============*/

void PrintMatrix(IntMatrixNxM matrix)
{
    for(uint32_t r = 0; r < matrix.RowCount; r++)
    {
        printf("( ");
        for(uint32_t c = 0; c < matrix.ColumnCount; c++)
            printf("%3i ", GetValueMatrix(matrix, r, c));
        printf(")\n");
    }
    printf("\n");
}