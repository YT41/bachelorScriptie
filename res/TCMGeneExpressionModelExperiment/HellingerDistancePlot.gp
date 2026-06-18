
set border lw 2
unset key
set xrange [0:3600]
set yrange [0:1]

set xlabel "t"
set ylabel "Hellinger Distance"

plot "hellingerDistance.data" using 1:2 w l lw 3
