
set border lw 2
unset key
set xrange [0:100]
set xlabel "t"
set ylabel "Species Count"
plot "trajectory.data" using 1:2:-2 w l lw 3 lc var
