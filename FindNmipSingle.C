#include <fstream>      
// std::filebuf
#define nMipsMax 3      
// what is the maximum number of MIPs you want to consider?
TF1* MipPeak[nMipsMax];
Double_t myfunc(Double_t* x, Double_t* param);  // Fit Function used by Minuit


void FindNmipSingle(int day=139){

  gStyle->SetOptStat(0);

  gStyle->SetTitleSize(0.2,"t");

  std::ofstream NmipFile(Form(
    "/mnt/d/27GeV/NmipSingle%dDay%d.txt",nMipsMax,day),ofstream::out);

  Float_t SingleMipPeakStartingValue,FitRangeLow,FitRangeHigh;
  FitRangeHigh               = 700.0;  // high edge of range along the x-axis
  Float_t xlo(50),xhi(1500);

  TF1* func = new TF1("MultiMipFit",myfunc,xlo,xhi,nMipsMax+2);

  // (1) ===================== Define the functions ========================
  MipPeak[0] = new TF1("1MIP","TMath::Landau(x,[0],[1],1)",xlo,xhi);
  for (Int_t nMIP=2; nMIP<=nMipsMax; nMIP++){
    TF1Convolution* c = new TF1Convolution(MipPeak[nMIP-2],MipPeak[0],
      xlo,xhi,true);
    MipPeak[nMIP-1] = new TF1(Form("%dMIPs",nMIP),c,xlo,xhi,2*nMIP);
  }

  // (2) ======================= Set up the fit ============================
  for (Int_t nmip=0; nmip<nMipsMax; nmip++){
    func->SetParName(nmip,Form("%dMIPweight",nmip+1));
  }
  func->SetParName(nMipsMax,"MPV");    
  func->SetParName(nMipsMax+1,"WIDbyMPV");  
  /// this is the Landau WID/MPV, and should be about 0.15 for the EPD
  func->SetParameter(nMipsMax+1,0.15);
  func->SetParLimits(nMipsMax+1,0.12,0.18);  
  /// usually I don't set limits, but this should be okay.
  /// ------------------------------------------------------------------ 
  func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
  func->SetLineWidth(2);


  TString EWstring[2] = {"East","West"};
  Float_t MaxPlot;
  TCanvas* theCanvas = new TCanvas("ADCs","ADCs",1200,700);
  TFile* in = new TFile(Form(
            "/mnt/d/27GeV/day%d.root",day),"READ");

    int ew=0;
    int PP=1;

    for (int i = 4; i < 5; ++i)
    {
    
    int TT=i+1;

    TPaveText* label = new TPaveText(0.2,0.3,0.8,0.9);
    label->AddText(Form("Day %d",day));
    label->AddText(Form("%s PP%2d",EWstring[ew].Data(),PP));
    label->Draw();

    TH1D* adc = (TH1D*)in->Get(Form("ADCEW%dPP%dTT%d",ew,PP,TT));
    adc->SetTitle(Form("%s PP%02d TT%02d",EWstring[ew].Data(),PP,TT));
    adc->GetXaxis()->SetTitle("ADC");
    adc->GetXaxis()->SetLabelSize(0.08);
    adc->GetXaxis()->SetTitleSize(0.08);
    adc->GetXaxis()->SetTitleOffset(1); 
    adc->GetXaxis()->SetRangeUser(50,1500);
    adc->SetMaximum(adc->GetBinContent(adc->GetMaximumBin())*1.4);
    Double_t yMax = adc->GetBinContent(adc->GetMaximumBin())*1.4;
    adc->SetMinimum(0);

          if (TT<10){         // QT32C
            FitRangeLow=105;
            FitRangeHigh=1500;
            SingleMipPeakStartingValue=140;
            MaxPlot=800;
          }
          else{               // QT32B
            FitRangeLow=85;
            FitRangeHigh=1500;
            SingleMipPeakStartingValue=115;
            MaxPlot=700;
          }

          
    adc->GetXaxis()->SetRangeUser(0,MaxPlot);
    func->SetParameter(nMipsMax+1,0.15);
    func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
    int FitStatus = adc->Fit("MultiMipFit","","",FitRangeLow,FitRangeHigh);
    
    Float_t nMipFound = func->GetParameter(nMipsMax);
          /// Let's include some errors in here. sk
    Float_t nMipError = func->GetParError(nMipsMax);
    NmipFile << Form("%d \t%d \t%d \t%d \t%f \t%f",
      day,ew,PP,TT,nMipFound,nMipError);
    NmipFile << endl;
    std::cout << nMipFound << std::endl;
    
          for (int n=0; n < nMipsMax; n++){
            TH1D* temp = (TH1D*)(adc->Clone());
            temp->Clear();
            for (int ibin=1; ibin<temp->GetXaxis()->GetNbins(); ibin++){
                temp->SetBinContent(ibin,abs(func->
                  GetParameter(n))*(*MipPeak[n])(temp->
                  GetXaxis()->GetBinCenter(ibin)));
              }
            temp->SetLineWidth(0);
            temp->SetLineColor(kBlue);
            //temp->SetFillStyle(3001);
            temp->SetFillColorAlpha(kBlue,0.25);
            temp->Draw("same");
          }

    TLine* tnominal = new TLine(SingleMipPeakStartingValue,0,
                                SingleMipPeakStartingValue,
                                yMax);
    tnominal->SetLineColor(4);    
    tnominal->Draw();

    //theCanvas->SaveAs("/mnt/d/27GeV/ADC5Compare.root");
      
    theCanvas->SaveAs(
        Form("/mnt/d/27GeV/Day%dTile%dMips%d.pdf",
              day,TT,nMipsMax));
    }
    //label->Delete();
    //in->Close();
    NmipFile.close();
}



// ------------------------- here is the fitting function ----------------
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
// --------------------------------------------------------------------------