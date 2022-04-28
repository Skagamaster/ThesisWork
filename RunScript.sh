#!/bin/bash
cd /mnt/d/Macros
sudo cp -r RunAnalysisPar.sh /home/zandar/root61402/StPicoEvent1/RunAnalysisPar.sh
cd /home/zandar/root61402/StPicoEvent1
sudo dos2unix RunAnalysisPar.sh
./RunAnalysisPar.sh