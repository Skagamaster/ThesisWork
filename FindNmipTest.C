#include <fstream>      // std::filebuf

#define nMipsMax 3       // Set the maximum MIPs to consider.
TF1* MipPeak[nMipsMax];
Double_t myfunc(Double_t* x, Double_t* param);  // Fit function used by Minuit.
Int_t SetMax(TH1D* adc, int iter=10, int jump=15, int jump1=10);

void FindNmipTest(int day=112){

  gStyle->SetOptStat(0);
  gStyle->SetTitleSize(0.2,"t");

  // Put where you'd like to save the text file of 1st MIP MPVs.
  std::ofstream NmipFile(Form(
    "/mnt/d/14GeV/NmipConstantsTestDay%d.txt",day),ofstream::out);

  Float_t SingleMipPeakStartingValue,FitRangeLow,FitRangeHigh;
  FitRangeHigh               = 1500.0;  // High edge of range along the x-axis.
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
  /// This is the Landau WID/MPV, and should be about 0.15 for the EPD.
  func->SetParameter(nMipsMax+1,0.15);
  func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
  func->SetLineWidth(3);

  /// Loop over East/West, PP, and TT.
  TString EWstring[2] = {"East","West"};
  Float_t MaxPlot;
  TCanvas* theCanvas = new TCanvas("ADCs","ADCs",1400,2400);
  theCanvas->Divide(4,8);
  theCanvas->SaveAs(
        Form("/mnt/d/14GeV/ADCspectraTestDay%d.pdf[",day));
  TFile* in = new TFile(Form(
            "/mnt/d/14GeV/Day%d.root",day),"READ");

  for (int ew=0; ew<1; ew++){
    for (int PP=1; PP<2; PP++){
      int iPad=0;
      theCanvas->cd(++iPad);
      TPaveText* label = new TPaveText(0.2,0.3,0.8,0.9);
      label->AddText(Form("Day %d",day));
      label->AddText(Form("%s PP%2d",EWstring[ew].Data(),PP));
      label->Draw();
        for (int TT=1; TT<2; TT++){
          cout << "EW:PP:TT = " << ew << ":" << PP << ":" << TT << endl;
          TPad* thePad = (TPad*)theCanvas->cd(++iPad);
        	thePad->SetTopMargin(0);
        	thePad->SetBottomMargin(0.2);
        	TH1D* adc = (TH1D*)in->Get(Form("AdcEW%dPP%dTT%d",ew,PP,TT));
        	adc->SetTitle(Form("%s PP%02d TT%02d",EWstring[ew].Data(),PP,TT));
        	adc->GetXaxis()->SetTitle("ADC");
        	adc->GetXaxis()->SetLabelSize(0.08);
        	adc->GetXaxis()->SetTitleSize(0.08);
        	adc->GetXaxis()->SetTitleOffset(1);
          adc->GetXaxis()->SetRangeUser(10,1500);
          //adc->SetMaximum(adc->GetBinContent(adc->GetMaximumBin())*1.4);
          adc->SetMinimum(0);
          
          // This finds the peak after the dark current.
          // The second argument is the starting value.
          // The third argument is the iteration amount to find the starting value.
          // The fourth argument is the iteration amount to find the ending value.
          Int_t Start, End = SetMax(adc, 10, 15, 10);
          //--------------------------------------------
          adc->SetMaximum(adc->GetBinContent(End)*1.7);
          FitRangeLow = Start + 0.6*(End-Start);
          SingleMipPeakStartingValue=End+10; // "Guess" for the 1st MIP MPV
          MaxPlot=SingleMipPeakStartingValue*5;
          adc->Draw();
        	adc->GetXaxis()->SetRangeUser(0,MaxPlot);
        	func->SetParameter(nMipsMax+1,0.15);
        	func->SetParameter(nMipsMax,SingleMipPeakStartingValue);

          /// This is the fit.
        	int FitStatus = adc->Fit("MultiMipFit","","",FitRangeLow,FitRangeHigh);
          Float_t nMipFound = func->GetParameter(nMipsMax);
          Float_t nMipError = func->GetParError(nMipsMax);
          cout << "FitStatus= " << FitStatus << endl;
          //-------------------------------------------------------------------
             /*
             This section is to handle fits that didn't work.
             Improper fits were either from a FitStatus not equal
             to 0, or because the expected, first MIP MPV was too
             far off from the found MPV. Both of these conditions
             must be met simultaneously for a satisfactory fit.

             The outside portion of the while loop handles the FitStatus != 0
             bit, and the inside while loop handles the MPV issue. It tries a
             new starting point to fix the FitStatus, and a new MPV to fix the
             MPV delta issue.
             */

          bool enderFit = 0;
          bool enderMPV = 0;
          bool enderFunc = 0;
          int maxDelta = 19; // This is what I'll have as an input.
          if (abs(nMipFound-SingleMipPeakStartingValue) > maxDelta)
          {
            enderMPV += 1;
          }
          if (FitStatus != 0)
          {
            enderFit += 1;
          }

          bool enderMaster = 0;
          while (((enderFit != 0) || (enderMPV != 0)) && (enderMaster == 0))
          {
            int counter = 1;
            while ((enderFit != 0) && (enderMaster ==0))
            {
              cout << "EW:PP:TT = " << ew << ":" << PP << ":" << TT << endl;
              cout << "Trying new starting point: " << counter << "/" << 1.6*End-0.4*Start+14 << endl;
              int StartPrime = Start + 5 + counter;
              FitStatus = adc->Fit("MultiMipFit","","",StartPrime,FitRangeHigh);
              nMipFound = func->GetParameter(nMipsMax);
              nMipError = func->GetParError(nMipsMax);
              counter += 1;
              cout << "FitStatus= " << FitStatus << endl;
              if (FitStatus == 0)
              {
                enderFit = 0;
              }
              if (StartPrime >= End)
              {
                if ((enderFit != 0) || (enderMPV != 0))
                {
                  enderMaster +=1;
                }
                break;
              }
              if (abs(nMipFound-SingleMipPeakStartingValue) > 19)
              {
                enderMPV += 1;
              }
              else
              {
                enderMPV = 0;
              }
              if ((enderMPV != 0) && (enderFit == 0))
              {
                int counter1 = 1;
                while (enderMPV != 0)
                {
                  cout << "Trying new MPV estimate " << counter1 << "/" << SingleMipPeakStartingValue + 10 - FitRangeLow
                  << " on position " << counter-1 << "/" << 1.6*End-0.4*Start+9 << "." << endl;
                  int SingleMipPeakStartingValuePrime = FitRangeLow - 1 + counter1;
                  func->SetParameter(nMipsMax,SingleMipPeakStartingValuePrime);
                  FitStatus = adc->Fit("MultiMipFit","","",StartPrime,FitRangeHigh);
                  nMipFound = func->GetParameter(nMipsMax);
                  nMipError = func->GetParError(nMipsMax);
                  counter1 += 1;
                  cout << "FitStatus= " << FitStatus << endl;
                  if (abs(nMipFound-SingleMipPeakStartingValuePrime) < 20)
                  {
                    enderMPV = 0;
                    if (FitStatus != 0)
                    {
                      enderFit += 1;
                    }
                    else
                    {
                      break;
                    }
                  }
                  if (SingleMipPeakStartingValuePrime >= End + 50)
                  {
                    break;
                  }
                }
              }
            }

            counter = 1;
            while ((enderMPV != 0) && (enderMaster == 0))
            {
              cout <<  "EW:PP:TT = " << ew << ":" << PP << ":" << TT << endl;
              cout << "Trying new MPV estimate: " << counter << "/" << End+51-FitRangeLow << endl;
              int SingleMipPeakStartingValuePrime = FitRangeLow-1+counter;
              func->SetParameter(nMipsMax,SingleMipPeakStartingValuePrime);
              FitStatus = adc->Fit("MultiMipFit","","",FitRangeLow,FitRangeHigh);
              nMipFound = func->GetParameter(nMipsMax);
              nMipError = func->GetParError(nMipsMax);
              SingleMipPeakStartingValuePrime += 1;
              counter += 1;
              cout << "FitStatus= " << FitStatus << endl;
              if (abs(nMipFound-SingleMipPeakStartingValuePrime) < 20)
              {
                enderMPV = 0;
              }
              if (SingleMipPeakStartingValuePrime > End + 50)
              {
                if ((enderMPV != 0) || (enderFit != 0))
                {
                  enderMaster += 1;
                }
                break;
              }
              if (FitStatus != 0)
              {
                enderFit += 1;
              }
              else
              {
                enderFit = 0;
              }
              if ((enderFit !=0) && (enderMPV == 0))
              {
                int counter1 = 1;
                while (enderFit != 0)
                {
                  cout << "EW:PP:TT = " << ew << ":" << PP << ":" << TT << endl;
                  cout << "Trying new starting point: " << counter1 << "/" << 1.6*End-0.4*Start+14 
                  << " on MPV " << counter-1 << "/" << End+51-FitRangeLow << endl;
                  int StartPrime = Start + 5 + counter1;
                  FitStatus = adc->Fit("MultiMipFit","","",StartPrime,FitRangeHigh);
                  nMipFound = func->GetParameter(nMipsMax);
                  nMipError = func->GetParError(nMipsMax);
                  counter1 += 1;
                  cout << "FitStatus= " << FitStatus << endl;
                  if (FitStatus == 0)
                  {
                    enderFit = 0;
                    if (abs(nMipFound-SingleMipPeakStartingValuePrime) > 19)
                    {
                      enderMPV += 1;
                    }
                    else
                    {
                      break;
                    }
                  }
                  if (StartPrime >= End)
                  {
                    break;
                  }
                }
              }
            }
          }
          
          /// These are to reset if the above fit retries failed.
          if (enderFit != 0)
          {
            FitRangeLow = Start + 0.6*(End-Start);
          }
          if (enderMPV != 0)
          {
            SingleMipPeakStartingValue=End+10;
          }          
//--------------------------------------------------------------------------------------------
          /// Last thing to try if the fit didn't work. This sets parameters for the fit that
          /// are reasonable, and sometimes will help with a particularly messy tile.
          
          float percent = 1.0;
          int counter = 1;
          while (((enderFit != 0) && (percent > 0.1)) || ((enderMPV != 0) && (percent > 0.1)))
          {
            cout << "EW:PP:TT = " << ew << ":" << PP << ":" << TT << endl;
            cout << "Trying new WID " << counter << "/10." << endl;
            FitRangeLow = Start + 0.6*(End-Start);
            func->SetParameter(nMipsMax,SingleMipPeakStartingValue);
            func->SetParLimits(nMipsMax+1,percent*0.13,percent*0.175);
            FitStatus = adc->Fit("MultiMipFit","","",FitRangeLow,FitRangeHigh);
            cout << "FitStatus= " << FitStatus << endl;
            nMipFound = func->GetParameter(nMipsMax);
            nMipError = func->GetParError(nMipsMax);
            percent -= 0.1;
            counter += 1;
            if (FitStatus == 0)
            {
              enderFit = 0;
            }
            if (abs(nMipFound-SingleMipPeakStartingValue) < 20)
            {
              enderMPV = 0;
            }
            if (FitStatus != 0)
            {
              enderFit += 1;
            }
            if (abs(nMipFound-SingleMipPeakStartingValue) > 19)
            {
              enderMPV += 1;
            }
          }
//###############################################################################################
          /// If none of the automatic fixes work, manually verify the Start and End positions
          /// and, if they're off, set them yourself. Same for the SingleMipPeakStartingValue.
          /// This should clear up a poor fit. Always inspect all fits visually even if they
          /// didn't have any errors when running this code.
//###############################################################################################
//-------------------------------------------------------------------------------------------
          adc->GetXaxis()->SetRangeUser(Start-20,SingleMipPeakStartingValue*3);
          adc->SetMaximum(adc->GetBinContent(nMipFound)*1.7);
          TLine* found = new TLine(nMipFound,0,nMipFound,adc->GetMaximum());
          found->SetLineColor(6);   found->Draw();
          /// Use these if you suspect the Start and End parameters aren't correct.
          //TLine* startLine = new TLine(Start,0,Start,adc->GetMaximum());
          //TLine* endLine = new TLine(End,0,End,adc->GetMaximum());
          //startLine->Draw();
          //endLine->Draw();

          /// Fill the text file with values.
          if ((enderFit !=0) || (enderMPV != 0))
          {
            thePad->SetFrameFillColor(kYellow-9);
            NmipFile << Form("%d \t%d \t%d \t%d \t",day,ew,PP,TT) << "######REFIT THIS TILE######";
            NmipFile << endl;
            cout << "#############FIT FAILED FOR " << ew << ":" << PP << ":" << TT << ".##############" << endl;
            if (FitStatus != 0)
            {
              cout << "FitStatus != 0; failed fit." << endl;
            }
            else
            {
              cout << "MPV delta too large for " << ew << ":" << PP << ":" << TT << "." << endl;
              cout << "Expected: " << SingleMipPeakStartingValue << endl;
              cout << "MPV was: " << nMipFound << endl;
            }
          }
          else
          {
            thePad->SetFrameFillColor(kWhite);
            NmipFile << Form("%d \t%d \t%d \t%d \t%f \t%f",day,ew,PP,TT,nMipFound,nMipError);
            NmipFile << endl;
          }          

          /// Display for single MIP fits.
          for (int n=0; n<nMipsMax; n++)
          {
            TH1D* temp = (TH1D*)(adc->Clone());
            temp->Clear();
            for (int ibin=1; ibin<temp->GetXaxis()->GetNbins(); ibin++)
            {
              temp->SetBinContent(ibin,abs(func->GetParameter(n))*(*MipPeak[n])(temp->GetXaxis()->GetBinCenter(ibin)));
            }
            temp->SetLineWidth(0);
            temp->SetLineColor(1);
            temp->SetFillStyle(1001);
            temp->SetFillColorAlpha(n+1,0.35);
            temp->Draw("hist lf2 same");
          }
        }
      theCanvas->SaveAs(
        Form("/mnt/d/14GeV/ADCspectraTestDay%d.pdf",day));
      label->Delete();
    }
  }
  in->Close();
  theCanvas->SaveAs(
    Form("/mnt/d/14GeV/ADCspectraTestDay%d.pdf]",day));
  NmipFile.close();
}
//--------------------------------------------------------------------------------
// Below are the functions for automatically finding the first peak.
//--------------------------------------------------------------------------------

/// This function finds the range for starting the fit (to exclude the dark
/// current). It also will attempt to find the first peak after the dark current.
/// These parameters are used to set the fit start range, SingleMipPeakStartingValue,
/// and to adjust the fit if it doesn't work on the first attmept.

Int_t SetMax(TH1D* adc, int iter=10, int jump=15, int jump1=10){
double First = 1.0;
double Second = 0.0;
double jumpD = 1.0*jump;
double jump1D = 1.0*jump1;
bool foundItS = 0;
bool foundItE = 0;
int iterate = 0;
int terminate = 0;
int begin = iter;

while(foundItS==0)
{
  while(First > Second)
 {
    First = 0.0;
    Second = 0.0;

    for(int i=iter; i<iter+jump; i++)
    {
      First += adc->GetBinContent(i);
      Second += adc->GetBinContent(i+jump);
    }
    First = First/jumpD;
    Second = Second/jumpD;
    iter += jump;
  }
  int iter1 = iter-jump;
  float First1 = 0.0;
  float Second1 = 0.0;
  for (int j = 0; j < 2; ++j)
  {
    for (int i = iter; i < iter1+jump; ++i)
    {
      First += adc->GetBinContent(i);
      Second += adc->GetBinContent(i+jump);
    }
    First1 = First1/jumpD;
    Second1 = Second1/jumpD;
  }
  if (Second > First)
  {
    foundItS += 1;
  }
  iterate += 1;
  if (iterate > 50)
  {
    iter = begin-5;
  }
  if (terminate > 100)
  {
    break;
  }
  terminate += 1;
}
int Start = iter-jump;

iterate = 0;
terminate = 0;

while(foundItE==0)
{
  while(Second > First)
  {
    First = 0.0;
    Second = 0.0;

    for(int i=iter; i<iter+jump1; i++)
    {
      First += adc->GetBinContent(i);
      Second += adc->GetBinContent(i+jump1);
    }
    First = First/jump1D;
    Second = Second/jump1D;
    iter += jump1;
  }
  int iter1 = iter;
  float First1 = 0.0;
  float Second1 = 0.0;
  for (int j = 0; j < 2; ++j)
  {
    for (int i = iter; i < iter1+jump1; ++i)
    {
      First += adc->GetBinContent(i);
      Second += adc->GetBinContent(i+jump1);
    }
    First1 = First1/jump1D;
    Second1 = Second1/jump1D;
  }
  if (First > Second)
  {
    foundItE += 1;
  }
  if (iterate > 50)
  {
    iter = Start;
  }
  if (terminate > 100)
  {
    break;
  }
  terminate += 1;
}

int End = iter;

return Start, End;
}


// ------------------------------- here is the fitting function -----------------------------
Double_t myfunc(Double_t* x, Double_t* param){
  // parameters 0...(nMipsMax-1) are the weights of the N-MIP peaks
  // and the last two parameters, index nMipsMax and nMipsMax+1,
  // are single-MIP MPV and WID/MPV, respectively
  Double_t ADC = x[0];
  Double_t SingleMipMPV = abs(param[nMipsMax]);
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
