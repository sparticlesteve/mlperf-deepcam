#!/bin/bash


echo "file workers seconds"

for logFile in $@; do
    numWorkers=$(grep "Number of workers" $logFile | awk '{print $4}')
    startTime=$(grep "run_start" $logFile | awk '{print $5}' | tr -d ',')
    stopTime=$(grep "run_stop" $logFile | awk '{print $5}' | tr -d ',')
    elapsedTime=$(echo "scale=3;($stopTime - $startTime)/1000" | bc -l)

    echo "$logFile $numWorkers $elapsedTime"
done
