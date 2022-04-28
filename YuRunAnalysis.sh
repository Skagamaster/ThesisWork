#!/bin/bash

for file in /mnt/d/14GeV/Full_Picos/*
do
    root -q -l -b "YuRunAnalysis.C("\""$file"\"")"
done

# Don't forget to: chmod +x
# If macros won't run in Linux, try the following:
# dos2unix