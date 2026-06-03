
set border lw 2
unset key
set xrange [0:]
set yrange [-0.002:0.002]

set xlabel "Epoch"
set ylabel "Loss"

plot "Loss.data" using 1:2 w d lw 3
