#!/bin/bash
    for i in /mnt/e/2021_Picos/7.7GeV/*.root; do
        root -q -l -b "RunAnalysis.C("\""$i"\"")"
    done &

    wait

# Don't forget to: chmod +x RunAnalysis.sh.
# If macros won't run in Linux, try the following:
# dos2unix RunAnalysis.sh