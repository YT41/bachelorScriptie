#include "MemArena.h"

#include <string.h>


MemArena ReallocMemArena(MemArena arena, size_t newAllocByteSize)
{
    uint64_t memoryPosOffset = ((uint64_t)(arena.currentMemoryPos) - (uint64_t)(arena.baseMemoryPos));
    void* newAllocPointer = malloc(newAllocByteSize);
    memcpy(newAllocPointer, (arena.baseMemoryPos), memoryPosOffset);
    free(arena.baseMemoryPos);
    return (MemArena){ newAllocPointer, (void*)((uint8_t*)newAllocPointer + memoryPosOffset) };
}