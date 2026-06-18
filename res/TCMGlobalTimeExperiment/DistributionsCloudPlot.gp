set multiplot layout 1,2

set border lw 2

set size ratio 1

unset key

set tics out
set tics nomirror

set view map
#set dgrid3d
#set pm3d interpolate 0, 0

set xrange [0:100]
set yrange [0:20]
set zrange [0:]

set xlabel "t"
set ylabel "n"

set title "{/:Bold Gillespie trajectory simulation}\n{/:Bold probability distribution}"
plot "fullDistribution.data" index 0 using 3:1:2 with image
set title "{/:Bold TCM predicted probability distribution}"
plot "fullDistribution.data" index 1 using 3:1:2 with image

unset multiplot
