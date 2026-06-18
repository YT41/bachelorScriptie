set multiplot layout 2,2


set border lw 2

set size ratio 1

unset key

set tics out
set tics nomirror

set view map
#set dgrid3d
#set pm3d interpolate 0, 0

set xrange [0:7]
set yrange [0:19]
set xtics 5
set ytics 5

unset colorbox
set cbrange [0:0.06]

set lmargin 2
set bmargin 2

set label 1 "{/:Bold Gillespie}" at graph 0.5,1.1 center

set ylabel "X2" offset 0,0
unset xlabel
set origin 0.0, 0.5
set size 0.45, 0.45
plot "fullDistribution2times.data" index 0 using 1:2:3 with image

set label 1 "{/:Bold TCM}" at graph 0.5,1.1 center
set label 2 "{/:Bold t = 1}" at graph 1.2,0.5 center

unset ylabel
unset xlabel
set origin 0.35, 0.5
set size 0.45, 0.45
plot "fullDistribution2times.data" index 1 using 1:2:3 with image

unset title
unset label

set ylabel "X2" offset 0,0
set xlabel "X1" offset 0,0
set origin 0.0, 0.05
set size 0.45, 0.45
plot "fullDistribution2times.data" index 2 using 1:2:3 with image


set colorbox user origin 0.85,0.1 size 0.05,0.8
set label "{/:Bold t = 10}" at graph 1.2,0.5 center

unset ylabel
set xlabel "X1" offset 0,0
set origin 0.35, 0.05
set size 0.45, 0.45
plot "fullDistribution2times.data" index 3 using 1:2:3 with image




unset multiplot
