#include "ReactionParser.h"

#include "MemArena.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>


/*====================== PARSER HELPER FUNCTIONS ======================*/

static inline int EOFReached(char* start, char* current, size_t fileSize)
{
    return ((size_t)((void*)current - (void*)start) >= fileSize);
}

static inline size_t GetFileSize(FILE* file)
{
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    return fileSize;
}

static inline uint32_t ReadUint32(char** textBufferPointerItPointer)
{
    return strtoul(*textBufferPointerItPointer, textBufferPointerItPointer, 10);
}

static inline double ReadDouble(char** textBufferPointerItPointer)
{
    return strtod(*textBufferPointerItPointer, textBufferPointerItPointer);
}

static inline size_t ReduceTextBuffer(char* textBuffer, size_t fileSize, char* reducedTextBuffer, uint32_t* speciesCount, uint32_t* reactionCount)
{
    char* textBufferPointerIt = textBuffer; /*first character*/
    size_t reducedTextBufferSize = 0;
    while(!EOFReached(textBuffer, textBufferPointerIt, fileSize))
    {
        switch(*textBufferPointerIt) 
        {
            case ' ':
                break;
            case '\n':
                break;
            case '\t':
                break;
            case '/':
                if(*(textBufferPointerIt + 1) == '*')
                {
                    textBufferPointerIt = strstr(textBufferPointerIt, "*/");
                    if(textBufferPointerIt == NULL)
                        return reducedTextBufferSize;
                    else
                        textBufferPointerIt++;
                }
                break;
            case '=':
                reducedTextBuffer[reducedTextBufferSize] = (*textBufferPointerIt);
                reducedTextBufferSize++;
                (*speciesCount)++;
                break;
            case '>':
                reducedTextBuffer[reducedTextBufferSize] = (*textBufferPointerIt);
                reducedTextBufferSize++;
                (*reactionCount)++;
                break;
            default:
                reducedTextBuffer[reducedTextBufferSize] = (*textBufferPointerIt);
                reducedTextBufferSize++;
                break;
        }
        textBufferPointerIt++;
    }
    return reducedTextBufferSize;
}

static inline void GetLines(char* reducedTextBuffer, size_t reducedTextBufferSize, char** speciesLineArr, char** reactionLineArr)
{
    char* textBufferPointerIt = reducedTextBuffer; /*first character*/
    uint32_t speciesLineIt = 0;
    uint32_t reactionLineIt = 0;
    char* lineBegin = textBufferPointerIt;
    bool isSpeciesLine = false;
    while(!EOFReached(reducedTextBuffer, textBufferPointerIt, reducedTextBufferSize))
    {
        switch(*textBufferPointerIt) 
        {
            case '=':
                isSpeciesLine = true;
                break;
            case ';':
                if(isSpeciesLine)
                {
                    speciesLineArr[speciesLineIt] = lineBegin;
                    speciesLineIt++;
                }
                else
                {
                    reactionLineArr[reactionLineIt] = lineBegin;
                    reactionLineIt++;
                }
                isSpeciesLine = false;
                lineBegin = (textBufferPointerIt + 1);
                break;
        }
        textBufferPointerIt++;
    }
}

static inline void GetSpecies(char** speciesLineArr, Species* speciesArr, uint32_t speciesCount)
{
    for(uint32_t i = 0; i < speciesCount; i++)
    {
        char* nameStart = speciesLineArr[i];
        char* nameEnd = strchr(speciesLineArr[i], '=');

        uint32_t count = (nameEnd - nameStart) + 1;
        snprintf(speciesArr[i].name, count, "%s", nameStart);

        nameEnd++;
        speciesArr[i].initialCount = ReadUint32(&nameEnd);
    }
}

static inline void GetColumnVals(char* reactionLine, int32_t* columnVals, const Species* speciesArr, uint32_t speciesCount)
{
    for(uint32_t i = 0; i < speciesCount; i++)
        columnVals[i] = 0;

    char* reactionLineIt = reactionLine;
    while(((*reactionLineIt) != '-') && ((*reactionLineIt) != ';'))
    {
        uint32_t count = 1;
        if(((*reactionLineIt) >= '0') && ((*reactionLineIt) <= '9'))
            count = ReadUint32(&reactionLineIt);

        char name[32] = { 0 };
        uint32_t nameIt = 0;
        while(((*reactionLineIt) != '+') && ((*reactionLineIt) != '-') && ((*reactionLineIt) != ';'))
        {
            name[nameIt] = (*reactionLineIt);
            nameIt++;
            reactionLineIt++;
        }

        for(uint32_t i = 0; i < speciesCount; i++)
        {
            if(strcmp(speciesArr[i].name, name) == 0)
            {
                columnVals[i] += count;
                break;
            }
        }

        if((*reactionLineIt) == '+')
            reactionLineIt++;
    }
}

static inline void GetReactions(char** reactionLineArr, Reaction** reactionArr, Species* speciesArr, uint32_t reactionCount, uint32_t speciesCount)
{
    for(uint32_t k = 0; k < reactionCount; k++)
    {
        int32_t reactantColumnVals[speciesCount]; 
        GetColumnVals(reactionLineArr[k], reactantColumnVals, speciesArr, speciesCount);

        char* lineIt = strchr(reactionLineArr[k], '-') + 2;
        double reactionRate = ReadDouble(&lineIt);
        lineIt += 2;

        int32_t productColumnVals[speciesCount]; 
        GetColumnVals(lineIt, productColumnVals, speciesArr, speciesCount);

        reactionArr[k] = CreateReaction(reactionRate, speciesCount, reactantColumnVals, productColumnVals);
    }
}


/*====================== PARSER PUBLIC FUNCTIONS ======================*/

SRN* ParseSRN(const char* fileName)
{
    FILE* SRNFilePointer = fopen(fileName, "r");
    size_t fileSize = GetFileSize(SRNFilePointer);

    MemArena arena = CreateMemArena((sizeof(char) * fileSize * 2));

    char* textBuffer = (char*)MemArenaAlloc(&arena, (sizeof(char) * fileSize));
    fread(textBuffer, sizeof(char), fileSize, SRNFilePointer);

    char* reducedTextBuffer = (char*)MemArenaAlloc(&arena, (sizeof(char) * fileSize));

    uint32_t speciesCount = 0;
    uint32_t reactionCount = 0;
    size_t reducedTextBufferSize = ReduceTextBuffer(textBuffer, fileSize, reducedTextBuffer, &speciesCount, &reactionCount);
    reducedTextBuffer[reducedTextBufferSize] = '\0';

    printf("%s\nspecies count = %u\nreaction count = %u\n", reducedTextBuffer, speciesCount, reactionCount);

    char* speciesLineArr[speciesCount];
    char* reactionLineArr[reactionCount];
    GetLines(reducedTextBuffer, reducedTextBufferSize, speciesLineArr, reactionLineArr);

    Species species[speciesCount];
    GetSpecies(speciesLineArr, species, speciesCount);

    Reaction* reactions[reactionCount];
    GetReactions(reactionLineArr, reactions, species, reactionCount, speciesCount);

    SRN* srn = CreateSRN(reactionCount, reactions, species);

    for(uint32_t k = 0; k < reactionCount; k++)
        DeleteReaction(reactions[k]);
    DeleteMemArena(&arena);
    fclose(SRNFilePointer);

    return srn;
}