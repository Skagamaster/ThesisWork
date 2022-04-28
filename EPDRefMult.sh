#!/bin/bash
for i in /mnt/d/27gev_production/st_physics_*.root
    do
    	root -q -l -b "EPDRefMult.C("\""$i"\"")"
    done

    #for i in /mnt/d/27GeV_Picos/st_mtd_19143*.root; do
    #    root -q -l -b "EPDRefMult.C("\""$i"\"")"
    #done &

    #wait
#for (( j = 142; j < 148; j++ )); do
#    hadd /mnt/d/27GeV/Day$j.root /mnt/d/27GeV/Day$j/*.root
#done
# You always have to do chmod +x EPDRefMult.sh