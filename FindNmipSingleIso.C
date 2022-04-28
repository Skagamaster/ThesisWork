#include <fstream>      // std::filebuf

#define nMipsMax 3      // what is the maximum number of MIPs you want to consider?
TF1* MipPeak[nMipsMax];
Double_t myfunc(Double_t* x, Double_t* param);  // Fit Function used by Minuit


void FindNmipSingleIso(int day=123){

  gStyle->SetOptStat(0);

  gStyle->SetTitleSize(0.2,"t");

  std::ofstream NmipFile(Form(
    "/mnt/d/Isobar/NmipSingle%dDay%d.txt",nMipsMax,day),ofstream::out);

  Float_t SingleMipPeakStartingValue,FitRangeLow,FitRangeHigh;
  FitRangeHigh               = 700.0;  // high edge of range along the x-axis
  Float_t xlo(50),xhi(1500);

  TF1* func = new TF1("MultiMipFit",myfunc,xlo,xhi,nMipsMax+2);

  // (1) ===================== Define the functions ==============================
  MipPeak[0] = new TF1("1MIP","TMath::Landau(x,[0],[1],1)",xlo,xhi);
  for (Int_t nMIP=2; nMIP<=nMipsMax; nMIP++){
    TF1Convolution* c = new TF1Convolution(MipPeak[nMIP-2],MipPeak[0],xlo,xhi,true);
    MipPeak[nMIP-1] = new TF1(Form("%dMIPs",nMIP),c,xlo,xhi,2*nMIP);
  }

  // (2) ======================= Set up the fit ======================================
  for (Int_t nmip=0; nmip < nMipsMax; nmip++)
  {
    func->SetParName(nmip,Form("%dMIPweight",nmip+1));
  }

  func->SetParName(nMipsMax,"MPV");    
  func->SetParName(nMipsMax+1,"WIDbyMPV");  
  // this is the Landau WID/MPV, and should be about 0.15 for the EPD
  func->SetParameter(nMipsMax+1,0.15);
  func->SetParLimits(nMipsMax+1,0.13,0.175);  
  // usually I don't set limits, but this should be okay.
  // ------------------------------------------------------------------ 
  func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
  func->SetLineWidth(3);
  func->SetLineColor(4);

  TString EWstring[2] = {"East","West"};
  Float_t MaxPlot;
  TCanvas* theCanvas = new TCanvas("ADCs","ADCs",1200,700);
  TFile* in1 = new TFile(Form(
            "/mnt/d/Isobar/Day%d.root",day),"READ");
  TFile* in2 = new TFile("/mnt/d/27Gev/Day145.root","READ");

    int ew=1;
    int PP=7;
    for (int i=13; i < 14; ++i)
    {
      int TT = i+1;

    TPaveText* label = new TPaveText(0.2,0.3,0.8,0.9);
    label->AddText(Form("Day %d",day));
    label->AddText(Form("%s PP%2d",EWstring[ew].Data(),PP));
    label->Draw();
    
    TH1D* adc = (TH1D*)in1->Get(Form("AdcEW%dPP%dTT%d",ew,PP,TT));
    //TH1D* adc2 = (TH1D*)in2->Get(Form("AdcEW%dPP%dTT%d",ew,PP,TT));

    adc->SetTitle(Form("%s PP%02d TT%02d",EWstring[ew].Data(),PP,TT));
    adc->GetXaxis()->SetTitle("ADC");
    adc->GetXaxis()->SetLabelSize(0.08);
    adc->GetXaxis()->SetTitleSize(0.08);
    adc->GetXaxis()->SetTitleOffset(1); 
    adc->GetXaxis()->SetRangeUser(50,1500);
    double adcMax = adc->GetBinContent(adc->GetMaximumBin());
    adc->SetMaximum(adcMax*1.4);
    adc->SetMinimum(0.0);
    //adc->Scale(1.5/adcMax);

    /*adc2->SetTitle(Form("%s PP%02d TT%02d",EWstring[ew].Data(),PP,TT));
    adc2->GetXaxis()->SetTitle("ADC");
    adc2->GetXaxis()->SetLabelSize(0.08);
    adc2->GetXaxis()->SetTitleSize(0.08);
    adc2->GetXaxis()->SetTitleOffset(1); 
    adc2->GetXaxis()->SetRangeUser(50,1500);
    adc2->SetMaximum(adc2->GetBinContent(adc2->GetMaximumBin())*1.4);
    adc2->SetMinimum(0.0);*/

    if (TT<10){         // QT32C
        FitRangeLow=115;
        FitRangeHigh=400;
        SingleMipPeakStartingValue=160;
        MaxPlot=600;
        }
    else{               // QT32B
        FitRangeLow=85;
        FitRangeHigh=1500;
        SingleMipPeakStartingValue=115;
        MaxPlot=400;
        }
          
    adc->GetXaxis()->SetRangeUser(0,MaxPlot);
    //adc2->GetXaxis()->SetRangeUser(0,MaxPlot);
    func->SetParameter(nMipsMax+1,0.15);
    func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
    int FitStatus = adc->Fit("MultiMipFit","","",FitRangeLow,FitRangeHigh);
    //int FitStatus2 = adc2->Fit("MultiMipFit","","",FitRangeLow,FitRangeHigh);
    
    Float_t nMipFound = func->GetParameter(nMipsMax);
    /// Let's include some errors in here. sk
    Float_t nMipError = func->GetParError(nMipsMax);

    NmipFile << Form("%d \t%d \t%d \t%d \t%f \t%f",day,ew,PP,TT,nMipFound,nMipError);
    NmipFile << endl;
    
    for (int n=0; n < nMipsMax; n++)
    {
        TH1D* temp = (TH1D*)(adc->Clone());
       // TH1D* temp2 = (TH1D*)(adc2->Clone());
        temp->Clear();
       // temp2->Clear();
        for (int ibin=1; ibin<temp->GetXaxis()->GetNbins(); ibin++)
        {
            temp->SetBinContent(ibin,abs(func->GetParameter(n))
                  *(*MipPeak[n])(temp->GetXaxis()->GetBinCenter(ibin)));
       //     temp2->SetBinContent(ibin,abs(func->GetParameter(n))
       //           *(*MipPeak[n])(temp2->GetXaxis()->GetBinCenter(ibin)));
        }
        temp->SetLineWidth(0);
        //temp2->SetLineWidth(0);
        temp->SetLineColor(1);
        //temp2->SetLineColor(1);
        //temp->SetFillStyle(3001);
        //temp2->SetFillStyle(3001);
        temp->SetFillColorAlpha(n+1,0.25);
        //temp2->SetFillColorAlpha(n+3,0.35);
        temp->Draw("same");
        //temp2->Draw("same");
    }
      theCanvas->SaveAs(
        Form("/mnt/d/Isobar_27GeV/MipPeaks/IsoDay%dTile%dMips%d.pdf",
          day,TT,nMipsMax));
      //theCanvas->SaveAs("/mnt/d/Isobar/ADC2Compare.root");
      /*label->Delete();
      in1->Close();
      //in2->Close();
      NmipFile.close();*/
    }
}



// ------------------------------- here is the fitting function -----------------------------
Double_t myfunc(Double_t* x, Double_t* param){
  // parameters 0...(nMipsMax-1) are the weights of the N-MIP peaks
  // and the last two parameters, index nMipsMax and nMipsMax+1,
  // are single-MIP MPV and WID/MPV, respectively
  Double_t ADC = x[0];
  Double_t SingleMipMPV = param[nMipsMax];
  Double_t WID = SingleMipMPV*param[nMipsMax+1];
  Double_t fitval=0.0;
  for (Int_t nMip=0; nMip<nMipsMax; nMip++){
    Double_t Weight = abs(param[nMip]);
    for (Int_t imip=0; imip<2*(nMip+1); imip+=2){
      MipPeak[nMip]->SetParameter(imip,SingleMipMPV);
      MipPeak[nMip]->SetParameter(imip+1,WID);
    }
    fitval += Weight*(*MipPeak[nMip])(ADC);
  }
  return fitval;
}
// -------------------------------------------------------------------------------------------