#include <fstream>      // std::filebuf

#define nMipsMax 3       // what is the maximum number of MIPs you want to consider?
TF1* MipPeak[nMipsMax];
Double_t myfunc(Double_t* x, Double_t* param);  // Fit Function used by Minuit


void FindNmipNoFit(int day=96){

  gStyle->SetOptStat(0);

  gStyle->SetTitleSize(0.2,"t");

  /*std::ofstream NmipFile(Form(
    "/mnt/d/27GeV/NmipConstantsDay%d.txt",day),ofstream::out);*/

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
  for (Int_t nmip=0; nmip<nMipsMax; nmip++){
    func->SetParName(nmip,Form("%dMIPweight",nmip+1));
  }
  func->SetParName(nMipsMax,"MPV");    
  func->SetParName(nMipsMax+1,"WIDbyMPV");  
  // this is the Landau WID/MPV, and should be about 0.15 for the EPD
  func->SetParameter(nMipsMax+1,0.15);
  //func->SetParLimits(nMipsMax+1,0.13,0.175);  
  /// usually I don't set limits, but this should be okay.
  /// **We were having some issues running up against the limits,
  /// so I commented that portion out.** sk 
  func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
  func->SetLineWidth(2);

  // okay now time to loop over Days, and PP, and TT...

  TString EWstring[2] = {"East","West"};
  Float_t MaxPlot;
  TCanvas* theCanvas = new TCanvas("ADCs","ADCs",1400,2400);
  //theCanvas->Divide(4,8);
  theCanvas->SaveAs(
        Form("/mnt/d/NewADC/ADCspectraDay%d.pdf[",day));
  TFile* in = new TFile(Form(
            "/mnt/d/Isobar/Day%d.root",day),"READ");

    for (int ew=0; ew<2; ew++){
    for (int PP=1; PP<13; PP++){
      int iPad=0;
      //theCanvas->cd(++iPad);
      TPaveText* label = new TPaveText(0.2,0.3,0.8,0.9);
      label->AddText(Form("Day %d",day));
      label->AddText(Form("%s PP%2d",EWstring[ew].Data(),PP));
      label->Draw();
        for (int TT=1; TT<32; TT++){
          /*TPad* thePad = (TPad*)theCanvas->cd(++iPad);
          thePad->SetTopMargin(0);
          thePad->SetBottomMargin(0.2);*/
          TH1D* adc = (TH1D*)in->Get(Form("ADCEW%dPP%dTT%d",ew,PP,TT));
          adc->SetTitle(Form("%s PP%02d TT%02d",EWstring[ew].Data(),PP,TT));
          adc->GetXaxis()->SetTitle("ADC");
          adc->GetXaxis()->SetLabelSize(0.08);
          adc->GetXaxis()->SetTitleSize(0.08);
          adc->GetXaxis()->SetTitleOffset(1); 
          adc->GetXaxis()->SetRangeUser(50,1500);
          adc->SetMaximum(adc->GetBinContent(adc->GetMaximumBin())*1.4);
          adc->SetMinimum(0);
          if (TT<10){         // QT32C
            /// We found that increasing the range to 1500
            /// yeilded better fits.
            FitRangeLow=110;
            FitRangeHigh=1500;
            SingleMipPeakStartingValue=160;
            MaxPlot=1500;
          }
          else{               // QT32B
            FitRangeLow=85;
            FitRangeHigh=1500;
            SingleMipPeakStartingValue=115;
            MaxPlot=1500;
          }

          
          adc->GetXaxis()->SetRangeUser(0,MaxPlot);
          func->SetParameter(nMipsMax+1,0.15);
          func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
          int FitStatus = adc->Fit("MultiMipFit","","",FitRangeLow,FitRangeHigh);
          //int FitStatus = 0;

          /*TLine* tnominal = new TLine(SingleMipPeakStartingValue,0,SingleMipPeakStartingValue,
                            adc->GetMaximum());
          tnominal->SetLineColor(4);    
          tnominal->Draw();*/
          //Float_t nMipFound = func->GetParameter(nMipsMax);
          /// Let's include some errors in here. sk
         // Float_t nMipError = func->GetParError(nMipsMax);
          //NmipFile << Form("%d \t%d \t%d \t%d \t%f \t%f",day,ew,PP,TT,nMipFound,nMipError);
          //if (FitStatus!=0) NmipFile << "\t <---------------- Fit failed";
          /// I turned the following off as I am inspecting visually.
          /*else if (fabs(nMipFound-SingleMipPeakStartingValue)>15) NmipFile << 
                    "\t <---------- different from nominal";*/
          //NmipFile << endl;
          //TLine* found = new TLine(nMipFound,0,nMipFound,adc->GetMaximum());
          //found->SetLineColor(6);   found->Draw();
          //if (FitStatus!=0)thePad->SetFrameFillColor(kYellow-9);
          //else thePad->SetFrameFillColor(kWhite);
        }
      theCanvas->SaveAs(
        Form("/mnt/d/NewADC/ADCspectraDay%d.pdf",day));
      label->Delete();
    }
  }
  in->Close();
  theCanvas->SaveAs(
    Form("/mnt/d/NewADC/ADCspectraDay%d.pdf]",day));
  //NmipFile.close();
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