#!/bin/bash

    root -q -l -b 'PrashRunAnalysis.C("/mnt/d/27GeV_Picos/st_mtd_19142027_raw_3000002.picoDst.root")' &
    root -q -l -b 'PrashRunAnalysis.C("/mnt/d/27GeV_Picos/st_mtd_19142034_raw_1000002.picoDst.root")' &
    root -q -l -b 'PrashRunAnalysis.C("/mnt/d/27GeV_Picos/st_mtd_19142035_raw_5500004.picoDst.root")' &
    root -q -l -b 'PrashRunAnalysis.C("/mnt/d/27GeV_Picos/st_mtd_19142038_raw_5500004.picoDst.root")' &
    root -q -l -b 'PrashRunAnalysis.C("/mnt/d/27GeV_Picos/st_mtd_19142039_raw_5500005.picoDst.root")' &
    root -q -l -b 'PrashRunAnalysis.C("/mnt/d/27GeV_Picos/st_mtd_19142041_raw_3000002.picoDst.root")' &

wait

# Recall: chmod +x RunTest.sh

