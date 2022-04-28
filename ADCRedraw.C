/// C++ Headers
#include <iostream>
#include <fstream>

/// ROOT Headers
#include "TFile.h"
#include "TChain.h"
#include "TSystem.h"
#include "TH1.h"
#include "TMath.h"
#include "TTree.h"

void ADCRedraw(int run=20110001){

  gStyle->SetOptStat(0);

  gStyle->SetTitleSize(0.2,"t");

  //std::ofstream NmipFile(Form(
  //  "PedestalsRun%d.txt",run),ofstream::out);

  // okay now time to loop over Days, and PP, and TT...

  TString EWstring[2] = {"East","West"};
  Float_t MaxPlot;
  TFile* in = new TFile(Form(
            "/mnt/d/14GeV/Day112/Runs/%d.root",run),"READ");

  TFile *MyFile = TFile::Open(Form("/mnt/d/14GeV/Day112/Runs/%drescale.root",run),"RECREATE");

  for (int ew=0; ew<2; ew++){
    for (int PP=1; PP<13; PP++){
      for (int TT=1; TT<32; TT++){
        TH1D* adc = (TH1D*)in->Get(Form("AdcEW%dPP%dTT%d",ew,PP,TT));
      	adc->SetTitle(Form("%s PP%02d TT%02d",EWstring[ew].Data(),PP,TT));
      	adc->GetXaxis()->SetTitle("ADC");

        int theMax = adc->GetXaxis()->GetBinLowEdge(adc->GetMaximumBin());
      	
        TH1D *newADC = new TH1D(Form("AdcEW%dPP%dTT%d",ew,PP,TT),Form("AdcEW%dPP%dTT%d",ew,PP,TT),1500,0,1500);

        for (Int_t i=0;i<1500;i++){
          newADC->SetBinContent(i,adc->GetBinContent(i+theMax));
        }

        newADC->Write();
      }
    }
  }
  MyFile->Close();
  //NmipFile.close();
}

