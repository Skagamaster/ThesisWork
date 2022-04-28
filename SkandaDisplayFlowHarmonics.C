/**
  This file is used to calculate flow harmonics
**/
// This is needed for calling standalone classes (not needed on RACF)
#define _VANILLA_ROOT_

// C++ headers
#include <iostream>
#include <sstream>

// ROOT headers
#include "TROOT.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TFile.h"
#include "THStack.h"
#include "TLegend.h"
#include "TChain.h"
#include "TTree.h"
#include "TSystem.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TMath.h"

double etaFromBin(int etaId);
int etaHistoBin(int etaId);

void SkandaDisplayFlowHarmonics(TString infile = "SkandaFlowCalculations_OUTPUT.root") { //include the name of the file i want to run on

    TFile *fInput = TFile::Open(infile);

    //_____________________________________________________________
    // Input histograms

    // Resolution
    TH1F* hRes1 = (TH1F*)fInput->Get("hRes1");
    TH1F* hRes2 = (TH1F*)fInput->Get("hRes2");
    TH1F* hResCent1[9];
    TH1F* hResCent2[9];
    for (int i = 0; i < 9; i++) {
        hResCent1[i] = (TH1F*)fInput->Get(Form("hResCent1_%d",i));
        hResCent2[i] = (TH1F*)fInput->Get(Form("hResCent2_%d",i));
    }

    // Histograms to be used to calculate flow harmonics from event plane across centrality
    TH1F* hCosPhiPsiCent1[2][9];            // Particles from EPD
    TH1F* hCosPhiPsiCent2[2][9];            // Particles from TPC
    TH1F* hWeightsCentTPC[2][9];
    TH1F* hWeightsCentEPD[2][9];
    for (int ew = 0; ew < 2; ew++) {
        for (int i = 0; i < 9; i++) {
            hCosPhiPsiCent1[ew][i] = (TH1F*)fInput->Get(Form("hCosPhiPsiCent1_%d_%d",ew,i));
            hCosPhiPsiCent2[ew][i] = (TH1F*)fInput->Get(Form("hCosPhiPsiCent2_%d_%d",ew,i));
            hWeightsCentTPC[ew][i] = (TH1F*)fInput->Get(Form("hWeightsCentTPC_%d_%d",ew,i));
            hWeightsCentEPD[ew][i] = (TH1F*)fInput->Get(Form("hWeightsCentEPD_%d_%d",ew,i));
        }
    }

    // Histograms to be used to calculate flow harmonics from event plane across eta
    TH1F* hCosPhiPsiEta1[9][20];
    TH1F* hCosPhiPsiEta2[9][20];
    TH1F* hWeightsEta[9][20];

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 20; j++) {
            hCosPhiPsiEta1[i][j] = (TH1F*)fInput->Get(Form("hCosPhiPsiEta1_%d_%d",i,j));
            hCosPhiPsiEta2[i][j] = (TH1F*)fInput->Get(Form("hCosPhiPsiEta2_%d_%d",i,j));
            hWeightsEta[i][j] = (TH1F*)fInput->Get(Form("hWeightsEta_%d_%d",i,j));
        }
    }

    // Histograms to be used to calculate flow harmonics from event plane across momentum
    TH1F* hCosPhiPsiPt1[2][9][10];
    TH1F* hCosPhiPsiPt2[2][9][10];
    TH1F* hWeightsPt[2][9][10];

    for (int ew = 0; ew < 2; ew++) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 10; j++) {
                hCosPhiPsiPt1[ew][i][j] = (TH1F*)fInput->Get(Form("hCosPhiPsiPt1_%d_%d_%d",ew,i,j));
                hCosPhiPsiPt2[ew][i][j] = (TH1F*)fInput->Get(Form("hCosPhiPsiPt2_%d_%d_%d",ew,i,j));
                hWeightsPt[ew][i][j] = (TH1F*)fInput->Get(Form("hWeightsPt_%d_%d_%d",ew,i,j));
            }
        }
    }

    //______________________________________________________________________
    // Histograms for displaying v_n{EP}

    double cBins[] = {0,5,10,20,30,40,50,60,70,80};
    double eBins[] = {-5.1,-4.6,-4.1,-3.6,-3.1,-2.6,-2.1,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,2.1,2.6,3.1,3.6,4.1,4.6,5.1};
    double pBins[] = {0,0.24,0.29,0.32,0.37,0.42,0.49,0.59,0.76,0.94,2};

    // Calculate resolution across centrality
    TH1F* hCentRes1 = new TH1F("hCentRes1",
        "Resolution of first-order event plane across centrality; %Centrality; R",
        9,cBins);
    TH1F* hCentRes2 = new TH1F("hCentRes2",
        "Resolution of second-order event plane across centrality; %Centrality; R",
        9,cBins);
    double res1[9];
    double resErr1[9];
    double res2[9];
    double resErr2[9];
    for (int centId = 0; centId < 9; centId++) {
        double resSq1 = hResCent1[centId]->GetMean();
        double resSq2 = hResCent2[centId]->GetMean();
        double resSqDev1 = hResCent1[centId]->GetStdDev();
        double resSqDev2 = hResCent2[centId]->GetStdDev();
        double resN1 = hResCent1[centId]->GetEntries();
        double resN2 = hResCent2[centId]->GetEntries();

        res1[centId] = sqrt(resSq1);
        res2[centId] = sqrt(resSq2);

        resErr1[centId] = 1/sqrt(resSq1*resN1)*resSqDev1;
        resErr2[centId] = 1/sqrt(resSq2*resN2)*resSqDev2;

        hCentRes1->Fill( cBins[8-centId], res1[centId] );
        hCentRes2->Fill( cBins[8-centId], res2[centId] );

        hCentRes1->SetBinError( 9-centId, resErr1[centId] );
        hCentRes2->SetBinError( 9-centId, resErr2[centId] );
    }
    double resSqAllCent1 = hRes1->GetMean();
    double resSqAllCent2 = hRes2->GetMean();
    double resSqAllCentDev1 = hRes1->GetStdDev();
    double resSqAllCentDev2 = hRes2->GetStdDev();
    double resAllCentN1 = hRes1->GetEntries();
    double resAllCentN2 = hRes2->GetEntries();

    double resAllCent1 = sqrt(resSqAllCent1);
    double resErrAllCent1 = resSqAllCentDev1/sqrt(resSqAllCent1*resAllCentN1);
    double resAllCent2 = sqrt(resSqAllCent2);
    double resErrAllCent2 = resSqAllCentDev2/sqrt(resSqAllCent2*resAllCentN2);

    // v_n across centrality
    TH1F* hCentV1 = new TH1F("hCentV1",
        "|v_{1}{EP}| across centrality in forward rapidity (2.1 < |#eta| < 5.1); %Centrality; v_{1}",
        9,cBins);
    TH1F* hCentV2 = new TH1F("hCentV2",
        "v_{2}{EP} across centrality in mid-rapidity (|#eta| < 1); %Centrality; v_{2}",
        9,cBins);
    for (int centId = 0; centId < 9; centId++) {
        double vEW1[2];
        double vEW2[2];
        double vEWErr1[2];
        double vEWErr2[2];
        for (int ew = 0; ew < 2; ew++) {
            double aveCosPhiPsi1 = hCosPhiPsiCent1[ew][centId]->GetMean();
            double aveCosPhiPsi2 = hCosPhiPsiCent2[ew][centId]->GetMean();
            double devCosPhiPsi1 = hCosPhiPsiCent1[ew][centId]->GetStdDev();
            double devCosPhiPsi2 = hCosPhiPsiCent2[ew][centId]->GetStdDev();
            double NCosPhiPsi1 = hCosPhiPsiCent1[ew][centId]->GetEntries();
            double NCosPhiPsi2 = hCosPhiPsiCent2[ew][centId]->GetEntries();

            double aveWTPC = hWeightsCentTPC[ew][centId]->GetMean();
            double aveWEPD = hWeightsCentEPD[ew][centId]->GetMean();

            vEW1[ew] = TMath::Abs(aveCosPhiPsi1/aveWEPD);
            vEW2[ew] = aveCosPhiPsi2/aveWTPC;

            vEWErr1[ew] = devCosPhiPsi1/aveWEPD/sqrt(NCosPhiPsi1);
            vEWErr2[ew] = devCosPhiPsi2/aveWTPC/sqrt(NCosPhiPsi2);
        }
        double v1 = (vEW1[0]+vEW1[1])/(2*res1[centId]);
        double v2 = (vEW2[0]+vEW2[1])/(2*res2[centId]);
        hCentV1->Fill( cBins[8-centId], v1 );
        hCentV2->Fill( cBins[8-centId], v2 );

        double vErr1 = sqrt( (pow(vEWErr1[0],2)+pow(vEWErr1[1],2))/4 + pow(v1*resErr1[centId],2) )/res1[centId];
        double vErr2 = sqrt( (pow(vEWErr2[0],2)+pow(vEWErr2[1],2))/4 + pow(v2*resErr2[centId],2) )/res2[centId];
        hCentV1->SetBinError(9-centId,vErr1);
        hCentV2->SetBinError(9-centId,vErr2);
    }

    TH1F* hEtaV1[9];
    TH1F* hEtaV2[9];
    for (int centId = 0; centId < 9; centId++) {
        hEtaV1[centId] = new TH1F(Form("hEtaV1_%d",centId),
            Form("v_{1}{EP} across #eta, %d-%d%% Centrality; #eta; v_{1}",(int)cBins[8-centId],(int)cBins[9-centId]),
            22,eBins);
        hEtaV2[centId] = new TH1F(Form("hEtaV2_%d",centId),
            Form("v_{2}{EP} across #eta, %d-%d%% Centrality; #eta; v_{2}",(int)cBins[8-centId],(int)cBins[9-centId]),
            22,eBins);

        for (int etaId = 0; etaId < 20; etaId++) {
            double aveCosPhiPsi1 = hCosPhiPsiEta1[centId][etaId]->GetMean();
            double aveCosPhiPsi2 = hCosPhiPsiEta2[centId][etaId]->GetMean();
            double devCosPhiPsi1 = hCosPhiPsiEta1[centId][etaId]->GetStdDev();
            double devCosPhiPsi2 = hCosPhiPsiEta2[centId][etaId]->GetStdDev();
            double NCosPhiPsi1 = hCosPhiPsiEta1[centId][etaId]->GetEntries();
            double NCosPhiPsi2 = hCosPhiPsiEta2[centId][etaId]->GetEntries();

            double aveW1 = hWeightsEta[centId][etaId]->GetMean();
            double aveW2 = hWeightsEta[centId][etaId]->GetMean();

            double vRaw1 = aveCosPhiPsi1/aveW1;
            double vRaw2 = aveCosPhiPsi2/aveW2;
            double vRawErr1 = devCosPhiPsi1/sqrt(NCosPhiPsi1)/aveW1;
            double vRawErr2 = devCosPhiPsi2/sqrt(NCosPhiPsi2)/aveW2;

            double eta = etaFromBin(etaId);
            double v1 = vRaw1/res1[centId];
            double v2 = vRaw2/res2[centId];
            hEtaV1[centId]->Fill( eta, v1 );
            hEtaV2[centId]->Fill( eta, v2 );

            int etaBin = etaHistoBin(etaId);
            double vErr1 = sqrt( pow(vRawErr1,2)+pow(v1*resErr1[centId],2) )/res1[centId];
            double vErr2 = sqrt( pow(vRawErr2,2)+pow(v2*resErr2[centId],2) )/res2[centId];
            hEtaV1[centId]->SetBinError( etaBin, vErr1 );
            hEtaV2[centId]->SetBinError( etaBin, vErr2 );
        }
    }

    // v_n across momentum
    TH1F* hPtV1 = new TH1F("hPtV1",
        "v_{1}{EP} across p_T; p_T (GeV/c); v_{1}",
        10,pBins);
    TH1F* hPtV2 = new TH1F("hPtV2",
        "v_{2}{EP} across p_T; p_T (GeV/c); v_{2}",
        10,pBins);
    TH1F* hCosPhiPsiPtAllCent1[2][10];
    TH1F* hCosPhiPsiPtAllCent2[2][10];
    TH1F* hWeightsPtAllCent[2][10];
    for (int ew = 0; ew < 2; ew++) {
        for (int pTId = 0; pTId < 10; pTId++) {
            hCosPhiPsiPtAllCent1[ew][pTId] = new TH1F(Form("hCosPhiPsiPt1_%d_%d",ew,pTId),
                Form("cos(#phi-#Psi_{1}) in pT bin %d, ew=%d",pTId,ew),
                300,-3.5,3.5);
            hCosPhiPsiPtAllCent2[ew][pTId] = new TH1F(Form("hCosPhiPsiPt2_%d_%d",ew,pTId),
                Form("cos(2*(#phi-#Psi_{2})) in pT bin %d, ew=%d",pTId,ew),
                200,-2,2);
            hWeightsPtAllCent[ew][pTId] = new TH1F(Form("hWeightsPt_%d_%d",ew,pTId),
                Form("Azimuthal track weights in pT bin %d, ew=%d",pTId,ew),
                200,0,3);
            for (int centId = 0; centId < 9; centId++) {
                hCosPhiPsiPtAllCent1[ew][pTId]->Add( hCosPhiPsiPt1[ew][centId][pTId] );
                hCosPhiPsiPtAllCent2[ew][pTId]->Add( hCosPhiPsiPt2[ew][centId][pTId] );
                hWeightsPtAllCent[ew][pTId]->Add( hWeightsPt[ew][centId][pTId] );
            }
        }
    }
    for (int pTId = 0; pTId < 10; pTId++) {
        double vEW1[2];
        double vEW2[2];
        double vEWErr1[2];
        double vEWErr2[2];
        for (int ew = 0; ew < 2; ew++) {
            double aveCosPhiPsi1 = hCosPhiPsiPtAllCent1[ew][pTId]->GetMean();
            double aveCosPhiPsi2 = hCosPhiPsiPtAllCent2[ew][pTId]->GetMean();
            double devCosPhiPsi1 = hCosPhiPsiPtAllCent1[ew][pTId]->GetStdDev();
            double devCosPhiPsi2 = hCosPhiPsiPtAllCent2[ew][pTId]->GetStdDev();
            double NCosPhiPsi1 = hCosPhiPsiPtAllCent1[ew][pTId]->GetEntries();
            double NCosPhiPsi2 = hCosPhiPsiPtAllCent2[ew][pTId]->GetEntries();

            double aveW = hWeightsPtAllCent[ew][pTId]->GetMean();

            vEW1[ew] = TMath::Abs(aveCosPhiPsi1/aveW);
            vEW2[ew] = aveCosPhiPsi2/aveW;

            vEWErr1[ew] = devCosPhiPsi1/aveW/sqrt(NCosPhiPsi1);
            vEWErr2[ew] = devCosPhiPsi2/aveW/sqrt(NCosPhiPsi2);
        }
        double v1 = (vEW1[0]+vEW1[1])/(2*resAllCent1);
        double v2 = (vEW2[0]+vEW2[1])/(2*resAllCent2);
        hPtV1->Fill( pBins[pTId], v1 );
        hPtV2->Fill( pBins[pTId], v2 );

        double vErr1 = sqrt( (pow(vEWErr1[0],2)+pow(vEWErr1[1],2))/4 + pow(v1*resErrAllCent1,2) )/resAllCent1;
        double vErr2 = sqrt( (pow(vEWErr2[0],2)+pow(vEWErr2[1],2))/4 + pow(v2*resErrAllCent2,2) )/resAllCent2;
        hPtV1->SetBinError(pTId+1,vErr1);
        hPtV2->SetBinError(pTId+1,vErr2);
    }

    // Writing to flow harmonic file
    TFile fFlowHarmonics("/mnt/d/27gev_production/data/SkandaFlowHarmonics.root","recreate");

    hCentRes1->Write();
    hCentRes2->Write();
    hCentV1->Write();
    hCentV2->Write();
    hPtV1->Write();
    hPtV2->Write();

    TCanvas* theCanvas = new TCanvas("v_{1,2}","Flow Parameters",1400,2400);
    theCanvas->Divide(3,3);
    int iPad = 0;

    for (int i = 0; i < 9; i++) {
        TPad* thePad = (TPad*)theCanvas->cd(++iPad);
        hEtaV1[i]->Write();
        hEtaV1[i]->Draw();
        //TPad* thePad1 = (TPad*)theCanvas->cd(++iPad);
        hEtaV2[i]->Write();
        //hEtaV2[i]->Draw();
    }

    theCanvas->SaveAs("/mnt/d/27gev_production/data/SkandaExample.pdf");

    fFlowHarmonics.Close();
    fInput->Close();
}



double etaFromBin(int etaId) {
    switch (etaId) {
        case 0: return -4.85;
        case 1: return -4.35;
        case 2: return -3.85;
        case 3: return -3.35;
        case 4: return -2.85;
        case 5: return -2.35;
        case 6: return -0.88;
        case 7: return -0.63;
        case 8: return -0.38;
        case 9: return -0.13;
        case 10: return 0.13;
        case 11: return 0.38;
        case 12: return 0.63;
        case 13: return 0.88;
        case 14: return 2.35;
        case 15: return 2.85;
        case 16: return 3.35;
        case 17: return 3.85;
        case 18: return 4.35;
        case 19: return 4.85;
        default: return -100;
    }
}



int etaHistoBin(int etaId) {
    if ( etaId<=5 ) return etaId+1;
    else if ( etaId<=13 ) return etaId+2;
    else return etaId+3;
}