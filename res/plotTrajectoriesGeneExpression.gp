
set multiplot layout 1,2

set border lw 2
unset key
set size ratio 1
set xrange [0:60]
set xlabel "t (Minutes)"

set ylabel "mRNA count"
plot "trajectory.data" using ($1 / 60):2:-2 w l lw 3 lc var

set ylabel "Protein Count"
plot "trajectory.data" using ($1 / 60):3:-2 w l lw 3 lc var

unset multiplot
