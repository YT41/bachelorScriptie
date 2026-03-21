#pragma once

#include <stdlib.h>
#include <stdint.h>


typedef struct MemArena
{
    void* baseMemoryPos;
    void* currentMemoryPos;
} MemArena;


static inline MemArena CreateMemArena(size_t allocByteSize)
{
    void* allocPointer = malloc(allocByteSize);
    return (MemArena){ allocPointer, allocPointer };
}

MemArena ReallocMemArena(MemArena arena, size_t newAllocByteSize);

static inline void DeleteMemArena(MemArena* arena)
{ 
    free(arena->baseMemoryPos);
}

static inline void* MemArenaAlloc(MemArena* arena, size_t allocByteSize)
{
    void* newAllocPointer = (arena->currentMemoryPos);
    arena->currentMemoryPos = (((uint8_t*)(arena->currentMemoryPos)) + allocByteSize);
    return newAllocPointer;
}