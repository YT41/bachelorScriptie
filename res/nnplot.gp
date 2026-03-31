
set border lw 2
unset key
set xrange [-2:2]

plot "NNTest.txt" using 1:2 w l lw 3, "NNTest.txt" using 1:3 w l lw 3


