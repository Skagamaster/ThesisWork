import os
import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import matplotlib.colors as colors

os.chdir(r"C:\Users\dansk\Documents\Thesis\AfterCuts_Initial")

Runs = np.load("Runs.npy", allow_pickle=True)
l = int(len(Runs))
x = np.linspace(0, l-1, l)  # For QA plots by run.

# Average Vr, Vz, and pT with error and 3 sigma deviations.
AveVr = np.load("AveVr.npy", allow_pickle=True)
stdVr = 3*np.std(AveVr[0])
aveVr = np.mean(AveVr[0])

AveVz = np.load("AveVz.npy", allow_pickle=True)
stdVz = 3*np.std(AveVz[0])
aveVz = np.mean(AveVz[0])

AvePt = np.load("AvePt.npy", allow_pickle=True)
stdPt = 3*np.std(AvePt[0])
avePt = np.mean(AvePt[0])

AveDca = np.load("AveDca.npy", allow_pickle=True)
stdDca = 3*np.std(AveDca[0])
aveDca = np.mean(AveDca[0])

AveRefMult3 = np.load("AveRefMult3.npy", allow_pickle=True)
stdRefMult3 = 3*np.std(AveRefMult3[0])
aveRefMult3 = np.mean(AveRefMult3[0])

dEdXpQ = np.load("dEdXpQ.npy", allow_pickle=True)
vR = np.load("vR.npy", allow_pickle=True)
vZ = np.load("vZ.npy", allow_pickle=True)
Runs = np.load("Runs.npy", allow_pickle=True)


def getBad(a, aAve, aStd):
    a = a[0]
    delta = abs(a-aAve)
    badRuns = np.hstack(np.asarray(np.where(delta > aStd)))
    return badRuns


badVr = Runs[getBad(AveVr, aveVr, stdVr)]
badVz = Runs[getBad(AveVz, aveVz, stdVz)]
badPt = Runs[getBad(AvePt, avePt, stdPt)]
badDca = Runs[getBad(AveDca, aveDca, stdDca)]
badRefMult3 = Runs[getBad(AveRefMult3, aveRefMult3, stdRefMult3)]

badRuns = np.unique(np.hstack((badVr, badVz, badPt, badDca, badRefMult3)))
goodRuns = np.setdiff1d(Runs, badRuns)
