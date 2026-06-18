
set border lw 2
unset key
set xrange [0:6]
set yrange [0:12]

set xlabel "Number of species in signaling cascade"
set ylabel "Runtime in hours"

plot "runtimes.data" using 1:2 w linespoints lw 2 pt 7
