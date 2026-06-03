set multiplot layout 1,2

set border lw 2

set size ratio 1

set xrange [0:]

set xlabel "t"

set yrange [0:15]
set ylabel "mean"
plot      'mean.data' index 1 w l lw 3 title 'BTCM', \
          'mean.data' index 0 w l lw 5 dt "..." title 'Gillespie'

unset yrange
set yrange[0:4]
set ylabel "standard deviation"
plot      'std.data' index 1 w l lw 3 title 'BTCM', \
          'std.data' index 0 w l lw 5 dt "..." title 'Gillespie'

unset multiplot
