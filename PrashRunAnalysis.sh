#!/bin/bash
for i in /mnt/d/14GeV/113/*; do
    root -q -l -b "RunAnalysis.C("\""$i"\"")";
done