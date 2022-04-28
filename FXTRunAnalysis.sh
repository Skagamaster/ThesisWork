#!/bin/bash
    for i in /mnt/d/FXT/27GeV/Picos/st_physics_19*.root; do
        root -q -l -b "FXTRunAnalysis.C("\""$i"\"")"
    done &
    for i in /mnt/d/FXT/27GeV/Picos/st_physics_adc_19*.root; do
        root -q -l -b "FXTADCRunAnalysis.C("\""$i"\"")"
    done &
    #for i in /mnt/d/14Gev/Picos/st_mtd_19166*.root; do
    #    root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    #done &
    #for i in /mnt/d/14Gev/Picos/st_mtd_19167*.root; do
    #    root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    #done &
    #for i in /mnt/d/14Gev/Picos/st_mtd_19168*.root; do
    #    root -q -l -b "PrashRunAnalysis.C("\""$i"\"")"
    #done &

    wait
#for (( j = 142; j < 148; j++ )); do
#    hadd /mnt/d/27GeV/Day$j.root /mnt/d/27GeV/Day$j/*.root
#done
# You always have to do chmod +x RunAnalysisPar.sh