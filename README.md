# Machine learning solutions to stochastic reaction networks

This repository contains all code that was used to perform the numerical experiments of my bachelor thesis, "Machine learning solutions to stochastic reaction networks". The thesis report can be found in the root directory of this repository. 

Every numerical experiment can be replicated, it is also quite easy to set up your own experiments. All the code was made from scratch, no external libraries are needed. The only program that you will have to install if you want to view the plots is gnuplot (http://www.gnuplot.info/). 

## documentation

Comments are provided for all the functions where it isn't immediately obvious what they do from the name, or are otherwise important to the core working of the project. The comments can be found in the header files (.hpp extension files), before the function declarations. Some comments have also been provided in function implementations where the code isn't obvious (can be found in .cpp extension files).

## building the project

If you are on linux (maybe this also works on a mac computer) then you can build the project by opening a console in the root directory of this repository and executing the following command:
`make release`

Then execute the code by executing the following command:
`make run`

If you are on Windows, open a powershell in the root directory of this repository (you can type "powershell" into the search bar). Then execute the following commands to build (you can just copy-paste everything into the console, that should work):
```
g++ -Wall -O4 -std=c++11 -c src/Attention.cpp -o obj/Attention.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/generalModelFunctions.cpp -o obj/generalModelFunctions.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/main.cpp -o obj/main.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/Matrix.cpp -o obj/Matrix.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/MemArena.cpp -o obj/MemArena.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/MiscMath.cpp -o obj/MiscMath.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/NeuralNetwork.cpp -o obj/NeuralNetwork.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/Random.cpp -o obj/Random.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/ReactionParser.cpp -o obj/ReactionParser.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/SRN.cpp -o obj/SRN.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/TCM.cpp -o obj/TCM.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11 -c src/TrajectorySim.cpp -o obj/TrajectorySim.o -I headers -I dependencies
g++ -Wall -O4 -std=c++11  obj/Attention.o  obj/generalModelFunctions.o  obj/main.o  obj/Matrix.o  obj/MemArena.o  obj/MiscMath.o  obj/NeuralNetwork.o  obj/Random.o  obj/ReactionParser.o  obj/SRN.o  obj/TCM.o  obj/TrajectorySim.o -o bin/main.exe -lm 
```

Then run the following command to execute the program:
`./bin/main.exe`

Alternatively, you can install make using chocolatey (https://chocolatey.org/), but then you might need to alter the makefile so that it works on windows.

## plotting

As mentioned, for plotting you need to install gnuplot (http://www.gnuplot.info/). All experiments have their own designated functions in the code and folders in /res. To perform an experiment, edit the main.cpp file by uncommenting all the experiment functions that you want to perform (some might take a while), then build and run the project as described earlier. In every folder you will find .gp files. open a console in the folder containing the .gp files, then you can run these scripts with the following general command:
`gnuplot --persist nameOfFile.gp`

(double-clicking might also work on windows)

## data

In /res you can find all sorts of extra saved plot images from previous experiments, organized into designated numbered folders. Most of these experiments did not make it into the thesis, some are of old versions of the model.

## importing your own SRNs

You can do this by either manually using the function declared in SRN.hpp, or you can make a human-readable file and import that using the "ParseSRN" function. The format that the "ParseSRN" function expects you to use goes as follows:

First declare the species of your SRN, example from the birth-death process SRN:
`B = 5 < 30;`

The first part before the "=" is the name of your species, "B" in this case. Then after the "=" it states the initial count of this species at time t = 0 (delta distribution). Then after the "<" is the truncation count N, as defined in the thesis in section 2.2. Then the line is ended with ";", like in C++, you need to always end lines with ";". You can add as many species as you like.

Now you can define the reaction of your SRN, example from the gene expression SRN:
`mRNA --0.1-> mRNA + protein;`

On the left side of the arrow you will find the reactants, on the right side you will find your products. Be sure to use the species names that you defined in previous lines. The number in the middle of the arrow is the reaction rate. The format of these lines follow the familiar general definition of reactions, see section 2.1 of the thesis. You can also add as many reactions as you like.

For a few examples, see "res/birthDeathModel.txt" and "res/GeneExpressionModel.txt".

## setting up your own experiment

I set up a template for custom experiments, see the function "TCMExperimentTemplate" in main.cpp. You can load any SRN that you want with the "ParseSRN" function (see "importing your own SRNs"). Be sure to set up a folder for your experiment and change the file name strings accordingly in the code. 

