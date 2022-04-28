#!/bin/bash
    for i in /mnt/d/2019Picos/9p2GeV/179/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/2019Picos/9p2GeV/183/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/2019Picos/9p2GeV/184/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/2019Picos/9p2GeV/185/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/2019Picos/9p2GeV/186/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/2019Picos/9p2GeV/187/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/2019Picos/9p2GeV/188/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/2019Picos/9p2GeV/189/*.root; do
        root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    done &

    wait
#for (( j = 142; j < 148; j++ )); do
#    hadd /mnt/d/27GeV/Day$j.root /mnt/d/27GeV/Day$j/*.root
#done
# You always have to do chmod +x RunAnalysisPar.sh