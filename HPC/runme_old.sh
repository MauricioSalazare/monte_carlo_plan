#!/bin/bash
source ../montecarlo/bin/activate >> log.txt
python python main.py -c 3 -n 10 >> log.txt
echo "Finished!!" >> log.txt











