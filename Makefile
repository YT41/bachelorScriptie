CC=gcc
FLAGS=-g -Wall -std=c11 -fsanitize=address
OBJ=obj
SRC=src
RES=res

#linux:
ADINCL=-I headers
LNK=-lm

RM=find
RMFLAGS=-type f -print -delete
CPY=cp
CPYFLAGS=-r


CPYDEST=bin/$(RES)

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SRCS=$(call rwildcard,$(SRC),*.c)
OBJS=$(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SRCS))
BIN=bin/main.exe


all: $(BIN)

gdb: $(BIN)
	gdb ./$(BIN)

run: $(BIN)
	./$(BIN)

cpyres:
	$(CPY) $(CPYFLAGS) $(RES) $(CPYDEST)

#normal release
release: FLAGS=-Wall -O2 -std=c11
release: clean
release: $(BIN)

#release but with profiler (about 15% slower)
profile: FLAGS=-Wall -O2 -std=c11 -pg
profile: clean
profile: $(BIN)

#plot the data
plot:
	cd res && gnuplot -persist plotTrajectoriesGeneExpression.gp

#for outputting the profiling results as a text file in the profiling folder
analyse:
	gprof $(BIN) gmon.out > "profiling/profilerLog.txt"


$(BIN): $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o $(BIN) $(LNK)

$(OBJ)/%.o: $(SRC)/%.c
	mkdir -p "$(dir $@)"
	$(CC) $(FLAGS) -c $< -o $@ $(ADINCL)


clean:
	cd $(OBJ) && $(RM) . -name "*.o" $(RMFLAGS)
	cd $(dir $(BIN)) && $(RM) . -name "*.exe" $(RMFLAGS)

exportflags: clean
exportflags: 
	bear -- make