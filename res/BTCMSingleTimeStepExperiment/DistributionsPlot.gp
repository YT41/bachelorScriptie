set multiplot layout 1,2

set border lw 2

set size ratio 1

unset key

set yrange [0:1]
set ytics 0, 0.1, 1
set xrange [0:20]

set style data histogram
set style histogram clustered gap 1
set boxwidth 1.2
set style fill solid 1.0

set xlabel "n"
set ylabel "Probability"

#TODO: maak legenda voor tijd ook
set title "{/:Bold BTCM predicted probability distribution at t = 1}"
plot "BTCMFullDistribution.data" using 2:xtic(5)
set title "{/:Bold Gillespie trajectory simulation}\n{/:Bold probability distribution at t = 1}"
plot "TrajectorySimFullDistribution.data" using 2:xtic(5)

unset multiplot
