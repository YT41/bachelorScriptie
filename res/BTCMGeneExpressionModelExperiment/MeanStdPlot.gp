set multiplot layout 1,2

set border lw 2

set size ratio 1

set xrange [0:3600]

set xlabel "t"

set yrange [0:70]
set ylabel "mean"
plot    'mean.data' index 1 using 1:2 w l lw 3 lc "red" title 'mRNA', \
        'mean.data' index 1 using 1:3 w l lw 3 lc "blue" title 'protein', \
        'mean.data' index 0 using 1:2 w l lw 5 dt "..." lc "red" notitle, \
        'mean.data' index 0 using 1:3 w l lw 5 dt "..." lc "blue" notitle

unset yrange
set yrange[0:]
set ylabel "standard deviation"
plot    'std.data' index 1 using 1:2 w l lw 3 lc "red" title 'mRNA', \
        'std.data' index 1 using 1:3 w l lw 3 lc "blue" title 'protein', \
        'std.data' index 0 using 1:2 w l lw 5 dt "..." lc "red" notitle, \
        'std.data' index 0 using 1:3 w l lw 5 dt "..." lc "blue" notitle

unset multiplot
