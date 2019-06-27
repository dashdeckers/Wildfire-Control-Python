# Do 10 Baseline runs
for (( i = 0; i < 10; i++ )); do
  python main.py -r -t Baseline -e 10000 -n "Baseline$i"
done
