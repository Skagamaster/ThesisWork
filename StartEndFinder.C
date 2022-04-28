//-----------------------------------------------------------------------------------
// This macro is to fit ADC spectra to a convoluted Landau function in order to
// calibrate the STAR Event Plane Detector. This macro finds the 1st MIP MPV for
// each of the EPD's 744 tiles, with error. It is designed to do so automatically,
// but there could be issues with that!
//
// If you find yourself getting a lot of, "REFIT THIS TILE" errors in the final text
// file, try playing with some of the parameters like where the macro starts looking
// for the peak, the step size for finding the peak, etc. If you only get a handful,
// then use the macro FindNmipFix.C to correct them.
//
// Right at the start, there's a parameter called nMipsMax. This is how many MIPs
// you want to consider. 3 is a good starting point for most energies, but if you're
// calibrating something like 200 GeV you might want to bump this to 5. You could
// realistically get away with 2 for anything 19.6 GeV or under, but 3 seems to 
// work pretty well even at lower energies.
//
// This macro is meant to be inexpensive for operator time, but at the tradeoff
// of being computationally expensive. It's best to run it over a single day or
// run, check to see how it works, tweak it if need be (the idea is that you won't
// have to at all), then run it over all your days/runs and let it do the work.
// As is often the case, the reality might be different than the intention ...
//
//
// The important parts of this code (the fitting function, for instance) were done
// by Mike Lisa of OSU. The automatic fitting portion was done by Skipper
// Kagamaster of Lehigh University. This code is currently maintained by Skipper
// Kagamaster; shoot me an email at skk317@lehigh.edu if you have any problems
// using it. Happy calibrating!
//
// -Skipper
// 
//-----------------------------------------------------------------------------------
#include <fstream>      // std::filebuf
std::pair<Int_t,Int_t> SetMax(TH1D* adc, int iter=10, int jump=10, int jump1=15);

void StartEndFinder(int day=113){

  //-----------------------------------------------------------------------------------
  // Here's where you'll input paths and can set some parameters for the fit.
  //
  // Enter the path where you want the end results to be saved.
  TString pathSave = Form("/mnt/d/14GeV/Day%d/Runs/RunPeds/",day);
  //
  // Enter the path where you'll get your .root file from (if different than above).
  TString pathLoad = pathSave;
  //
  // Finally, enter the structure for your .root file (mine here is, "Day123.root").
  // This will most likely be changed if you're running over days vs run #s.
  TString rootLoad = Form("Day%d.root",day);
  //
  // If the start/end points are really off, change parameters here:
  Int_t StartingPoint = 20; // Where we'll start the finding algorithm for the peak
  Int_t FirstIterator = 15; // How much to step by when finding the dip after dark current
  Int_t SecondIterator = 20; // How much to step by when finding the end of the peak
  //
  // This is the maximum you want to have the 1st MIP MPV guess and the found value to be apart.
  int maxDelta = 19;
  //
  // You shouldn't need to enter enything below this line.
  //-----------------------------------------------------------------------------------

  gStyle->SetOptStat(0);
  gStyle->SetTitleSize(0.2,"t");

  TString txtSave = Form("NmipConstantsDay%d.txt",day);
  TString pdfSaveTop = Form("ADCspectraDay%d.pdf[",day);
  TString pdfSave = Form("ADCspectraDay%d.pdf[",day);
  TString pdfSaveBot = Form("ADCspectraDay%d.pdf]",day);

  std::ofstream NmipFile(pathSave+txtSave,ofstream::out);

  Float_t SingleMipPeakStartingValue,FitRangeLow,FitRangeHigh;
  FitRangeHigh               = 1500.0;  // High edge of range along the x-axis.
  Float_t xlo(50),xhi(1500);


  /// Loop over East/West, PP, and TT.
  TString EWstring[2] = {"East","West"};
  Float_t MaxPlot;
  TCanvas* theCanvas = new TCanvas("ADCs","ADCs",1400,2400);
  theCanvas->Divide(4,8);
  theCanvas->SaveAs(pathSave+pdfSaveTop);
  TFile* in = new TFile(pathLoad+rootLoad,"READ");

  for (int ew=0; ew<2; ew++){
    for (int PP=1; PP<13; PP++){
      int iPad=0;
      theCanvas->cd(++iPad);
      TPaveText* label = new TPaveText(0.2,0.3,0.8,0.9);
      label->AddText(Form("Day %d",day));
      label->AddText(Form("%s PP%2d",EWstring[ew].Data(),PP));
      label->Draw();
        for (int TT=1; TT<32; TT++){
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
          adc->SetMinimum(0);
          
          // This finds the peak after the dark current.
          std::pair<Int_t,Int_t> p = SetMax(adc, StartingPoint, FirstIterator, SecondIterator);
          Int_t Start = p.first;
          Int_t End = p.second;
          //--------------------------------------------
          //adc->SetMaximum(adc->GetBinContent(End)*1.7);
          adc->SetMaximum(250);
          SingleMipPeakStartingValue=End;
          MaxPlot=SingleMipPeakStartingValue*5;
          //adc->Draw();
          adc->GetXaxis()->SetRangeUser(0,200);
//-------------------------------------------------------------------------------------------
          adc->Draw();
          //adc->GetXaxis()->SetRangeUser(Start-20,SingleMipPeakStartingValue*3);
          /// Use these if you suspect the Start and End parameters aren't correct.
          TLine* startLine = new TLine(Start,0,Start,adc->GetMaximum());
          TLine* endLine = new TLine(End,0,End,adc->GetMaximum());
          startLine->SetLineColor(2);
          startLine->Draw("same");
          endLine->SetLineColor(4);
          endLine->Draw("same");

        }
      theCanvas->SaveAs(pathSave+pdfSave);
      label->Delete();
    }
  }
  in->Close();
  theCanvas->SaveAs(pathSave+pdfSaveBot);
  NmipFile.close();
}
//--------------------------------------------------------------------------------
// Below are the functions for automatically finding the first peak.
//--------------------------------------------------------------------------------

/// This function finds the range for starting the fit (to exclude the dark
/// current). It also will attempt to find the first peak after the dark current.
/// These parameters are used to set the fit start range, SingleMipPeakStartingValue,
/// and to adjust the fit if it doesn't work on the first attmept.

std::pair<Int_t,Int_t> SetMax(TH1D* adc, int iter=10, int jump=10, int jump1=15){
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
    for (int i = iter1; i < iter1+jump; ++i)
    {
      First1 += adc->GetBinContent(i);
      Second1 += adc->GetBinContent(i+jump);
    }
    First1 = First1/jumpD;
    Second1 = Second1/jumpD;
  }
  if ((Second > First) && (Second1 > First1))
  {
    foundItS += 1;
  }
  iterate += 1;
  if (iterate > 50)
  {
    iter = begin+5;
  }
  if (terminate > 100)
  {
    cout << "TERMINATED" << endl;
    break;
  }
  terminate += 1;
}
int Start = iter;

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

int End = iter-jump1;

return std::make_pair(Start, End);
}

/*
std::pair<Int_t,Int_t> SetMax(TH1D* adc, int iter=10, int jump=10, int jump1=15){
double First = 0.0;
double Second = -1.0;
double jumpD = 1.0*jump;
double jump1D = 1.0*jump1;
bool foundItS = 0;
bool foundItE = 0;
int iterate = 0;
int terminate = 0;
int begin = iter;

while(foundItS==0)
{
  while((First || Second) < 0.0)
 {
    float slope1 = 0.0;
    float slope2 = 0.0;
    float slopeA = 0.0;
    float slopeB = 0.0;

    for (int i = iter; i < iter+jump; ++i)
    {
      slope1 += adc->GetBinContent(i);
      slope2 += adc->GetBinContent(i+jump);
      slopeA += adc->GetBinContent(i+int(jumpD/2));
      slopeB += adc->GetBinContent(i+int((3*jumpD)/2));
    }

    First = (slope2-slope1);
    Second = (slopeB-slopeA);
    iter += jump;

  }


  if ((First && Second) > 0.0)
  {
    foundItS += 1;
    cout << "BOO-YEAH!!!!" << endl;
  }
  iterate += 1;
  if ((iterate > 50) && (iter > 0.0))
  {
    iter = begin+jump;
  }
  if (terminate > 100)
  {
    break;
  }
  terminate += 1;
  First = 0.0;
  Second = -1.0;
}

int Start = iter;

iterate = 0;
terminate = 0;

while(foundItE==0)
{
  while((First || Second) > 0.0)
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
    float slope1e = 0.0;
    float slope2e = 0.0;
    float slopeAe = 0.0;
    float slopeBe = 0.0;

    for (int i = iter; i < iter+jump1; ++i)
    {
      slope1e += adc->GetBinContent(i);
      slope2e += adc->GetBinContent(i+int(jump/4));
      slopeAe += adc->GetBinContent(i+2*jump);
      slopeBe += adc->GetBinContent(i+2*jump+int(jump/4));
    }

    First = slope2e-slope1e/pow(jump1D,2);
    Second = slopeBe-slopeAe/pow(jump1D,2);
    iter += jump1;

  if ((First && Second) < 0.0)
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

int End = iter-jump1;

return std::make_pair(Start, End);
}*/