// This is needed for calling standalone classes (not needed on RACF)
#define _VANILLA_ROOT_

// C++ headers
#include <iostream>
#include <sstream>

// ROOT headers
#include "TROOT.h"
#include "TFile.h"
#include "TChain.h"
#include "TTree.h"
#include "TSystem.h"
#include "TH1.h"
#include "TH2.h"
#include "TMath.h"

// PicoDst headers
#include "StPicoDstReader.h"
#include "StPicoDst.h"
#include "StPicoEvent.h"
#include "StPicoTrack.h"
#include "StPicoBTofHit.h"
#include "StPicoBTowHit.h"
#include "StPicoEmcTrigger.h"
#include "StPicoBTofPidTraits.h"
//#include "StPicoTrackCovMatrix.h"

// Load libraries (for ROOT_VERSTION_CODE >= 393215)
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,0,0) 
R__LOAD_LIBRARY(libStPicoDst)
#endif

// inFile - is a name of name.picoDst.root file or a name
//          of a name.lis(t) files that contains a list of
//          name1.picoDst.root files

//_________________
void EPDRefMult(const Char_t *inFile = 
"/mnt/d/27gev_production/picos/st_physics_19130071_raw_1000005.picoDst.root") {

	StPicoDstReader* picoReader = new StPicoDstReader(inFile);
	picoReader->Init();

	picoReader->SetStatus("*",0);
 	picoReader->SetStatus("Event",1);
  picoReader->SetStatus("Track",1);
  picoReader->SetStatus("EpdHit",1);

  Long64_t events2read = picoReader->chain()->GetEntriesFast();
  Long64_t split = events2read/6.0;
  Long64_t split1 = 5.0*split;
  int train_event = (int)split1;
  //Long64_t events2read = 1000;

  std::cout << "Number of events to read: " << events2read << std::endl;
  cout << "Training events: " << train_event << endl;

  /// These are the histos for nMIP vs RefMult used for the polynomial fit.
  TH2D *hEPDRefMult[17];
  double nMIP[17];

  /// Histos for the event cuts.
  TH2D *hVr = new TH2D("hVr", "Vr Distribution;Vx;Vy",401,-0.5,0.5,401,-0.5,0.5);
  TH2D *hVrCut = new TH2D("hVrCut", "Vr Distribution after Cuts;Vx;Vy",401,-0.5,0.5,401,-0.5,0.5);
  TH1D *hVz = new TH1D("hVz","Vz Distribution;Vz;Counts",101,-0.5,0.5);
  TH1D *hVzCut = new TH1D("hVzCut","Vz Distribution after Cuts;Vz;Counts",101,-0.5,0.5);
  TH1D *hVzVPD = new TH1D("hVzVPD","|Vz-VzVPD|;|Vz-VzVPD|;Counts",101,0.0,30.0);
  TH1D *hVzVPDCut = new TH1D("hVzVPDCut","|Vz-VzVPD| after Cuts;|Vz-VzVPD|;Counts",101,0.0,30.0);
  TH1D *hBTOFMatch = new TH1D("hBTOFMatch","BTOFMatch;BTOFMatch;Counts",61,-1.0,30.0);
  TH1D *hBTOFMatchCut = new TH1D("hBTOFMatchCut","BTOFMatch after Cuts;BTOFMatch;Counts",61,-1.0,30.0);

  /// Declare the polynomial degree, weights, and nMIP max.
  /// Ring 1 will be double this number, ring 16 will be this number.
  double maxMIPset = 2.5;
  double maxMIP = maxMIPset;
  int polyFit = 3;
  double w[17];

  for (int i = 1; i < 17; ++i)
  {
    hEPDRefMult[i] = new TH2D(Form("hEPD%dRefMult",i),
      Form("Ring %i EPD;nMIP;RefMult",i),250,-0.5,119.5,250,-0.5,499.5);
    nMIP[i] = 0.0;
    // w[i] = 1.0/16.0; // No weight
    //w[i] = 0.0078431*(i); // Linear weight
    w[i] = 0.01172*pow(i,2); // Quadratic weight

    /// Quadratic with step function weight
    /*double a = 0.0030503;
    double b = 4*a;
    if (i > 6)
    {
      w[i] = b*pow(i,2);
    }
    else
    {
      w[i] = a*pow(i,2);
    }*/
    //hEPDRefMult[i]->Fill(nMIP[i],0.0);
  }
  hEPDRefMult[0] = new TH2D("hFullEPDRefMult","Full EPD;nMIP;RefMult",250,-0.5,1999.5,250,-0.5,499.5);

  ///How about a graph?
  TCanvas *c1 = new TCanvas("c1", "c1",475,84,700,502);
   c1->Range(-187.5,-0.5359811,1687.5,1.81353);
   c1->SetFillColor(0);
   c1->SetBorderMode(0);
   c1->SetBorderSize(2);
   //c1->SetLogy();
   //c1->SetLogx();
   c1->SetFrameBorderMode(0);
   c1->SetFrameBorderMode(0);

// Finding the day parameter.
  std::string Queso = inFile;
  std::string Nachos = "Oh no!";
  Queso = Queso.substr( Queso.find_last_of('/')+1 );
  if (Queso.substr(11,1) == "1")
  {
    Nachos = Queso.substr(13,3);
  }
  else
  {
    Nachos = Queso.substr(17,3);
  }
  int Tacos = stoi(Nachos);
  TString pathSave = "/mnt/d/27gev_production/data/";
  TFile *MyFile = TFile::Open(pathSave+Queso,"RECREATE");

  /// This file will be used to store the results from the polynomial fit.
  std::ofstream RefMultFile("/mnt/d/27gev_production/data/array_pol3.txt",
    ofstream::out);

  /// These files will be used to store the information on nMIP vs refMult for
  /// each of the rings, with ring0 being all together.
/*  ofstream RingFile[17];
  for (int i = 0; i < 17; ++i)
  {
    RingFile[i].open(Form("/mnt/d/27gev_production/data/ring%d.txt",i));
  }*/

/// Event loop
  for(Long64_t iEvent=0; iEvent<events2read; iEvent++) {
  //for(Long64_t iEvent=0; iEvent<2; iEvent++) {

    Bool_t readEvent = picoReader->readPicoEvent(iEvent);
    if( !readEvent ) {
      std::cout << "Error on event #" << iEvent << ". No data to chew on. :(" << std::endl;
      break;
    }

    /// Retrieve picoDst
    StPicoDst *dst = picoReader->picoDst();

    /// Retrieve event information
    StPicoEvent *event = dst->event();
    if( !event ) {
      std::cout << "Event #" << iEvent << " does not exist." << std::endl;
      break;
    }

    /// Fill the raw histos.
    hVr->Fill(event->primaryVertex().x(),event->primaryVertex().y());
    hVz->Fill(TMath::Abs(event->primaryVertex().z()));
    hVzVPD->Fill(TMath::Abs(event->vzVpd()-event->primaryVertex().z()));
    hBTOFMatch->Fill(event->nBTOFMatch());

    /// Some event cuts.
    if (TMath::Abs(event->vzVpd()-event->primaryVertex().z()) > 10.0 )
     {
       continue;
     }

    if (event->nBTOFMatch() < 0.00)
    {
      continue;
    }

     if (TMath::Abs(event->primaryVertex().z()) > 40.0 )
     {
      continue;
     }

     if (TMath::Abs( sqrt(pow(event->primaryVertex().x(),2) + 
      pow(event->primaryVertex().y(),2))) > 2.0 )
     {
       continue;
     }
    /// Fill the cuts histos.
    hVrCut->Fill(event->primaryVertex().x(),event->primaryVertex().y());
    hVzCut->Fill(TMath::Abs(event->primaryVertex().z()));
    hVzVPDCut->Fill(TMath::Abs(event->vzVpd()-event->primaryVertex().z()));
    hBTOFMatchCut->Fill(event->nBTOFMatch());

    /// EPD Hit loop

    TClonesArray *mEpdHits = dst->picoArray(8); 
    StPicoEpdHit* epdHit;

    for (int hit=0; hit<mEpdHits->GetEntries(); hit++){
      epdHit = (StPicoEpdHit*)((*mEpdHits)[hit]);
      double nMIPvalue = epdHit->nMIP();
      int tt = epdHit->tile();

      for (int k = 0; k < 32; k+=2)
      {

        int ring = (k/2)+1;
        if ((tt >= k) && (tt < k+2) && (nMIPvalue > 0.2) && (nMIPvalue <= maxMIP))
        {
          nMIP[ring] += nMIPvalue;
          nMIP[0] += w[ring]*nMIPvalue;
        }

        /// This loop will make the outer ring have a max nMIP
        /// = maxMIPset, and the innermost ring 2x that.
        else if ((tt >= k) && (tt < k+2) && (nMIPvalue > maxMIP) )
        {
          maxMIP = maxMIPset*(31.0/15.0-ring/15.0);
          nMIP[ring] += maxMIP;
          nMIP[0] += w[ring]*maxMIP;
        }

        else
          continue;
      }
        
    }

    RefMultFile << Form("%d \t",event->refMult()) << " " << nMIP[0] << endl;

    for (int j = 0; j < 17; ++j)
    {
      if (iEvent < train_event)
      {
        hEPDRefMult[j]->Fill(nMIP[j], event->refMult());
      }
      //hEPDRefMult[j]->Fill(nMIP[j], event->refMult());
      //RingFile[j] << Form("%f \t%d",nMIP[j],event->refMult()) << endl;
      /*if (j != 0)
      {
        RefMultFile << Form("%f \t",nMIP[j]) << endl;
      }*/
      nMIP[j] = 0.0;
    }


}

/// Here are the fit functions and graphs for a polynomial fit.

  /// pol3 fit for all rings
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf[");
hVz->Draw();
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hVzCut->Draw();
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hVr->Draw("colz");
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hVrCut->Draw("colz");
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hVzVPD->Draw();
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hVzVPDCut->Draw();
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hBTOFMatch->Draw();
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hBTOFMatchCut->Draw();
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hEPDRefMult[0]->SetStats(kFALSE);
hEPDRefMult[0]->Draw("colz");
TF1 *f = ((TF1*)(gROOT->GetFunction(Form("pol%d", polyFit))));
/// Use these to set polyn params if needed.
//f->SetParameter(0,0.0);
//f->SetParameter(1,0.5); // etc.
/// This guarantees the function passes through the origin.
f->FixParameter(0,0.);
f->SetParameter(2,0.);
TFitResultPtr r = hEPDRefMult[0]->Fit(f, "S");
TMatrixDSym cov = r->GetCovarianceMatrix();
/*for (int j = 0; j < polyFit+1; ++j)
{
  //cout.setf(ios::scientific);
  RefMultFile<<std::scientific;
  RefMultFile<<Form("%f \t",r->Value(j));
  RefMultFile<<endl;
}*/
c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
hEPDRefMult[0]->Write();
//RingFile[i].close();
  for (int i = 1; i < 17; ++i)
  {
    hEPDRefMult[i]->SetStats(kFALSE);
    hEPDRefMult[i]->Draw("colz");
    //TF1 *f = ((TF1*)(gROOT->GetFunction(Form("pol%d", polyFit))));
    /// Use these to set polyn params if needed.
    //f->SetParameter(0,0.0);
    //f->SetParameter(1,0.5); // etc.
    /// This guarantees the function passes through the origin.
    //f->FixParameter(0,0.);
    //f->SetParameter(2,0.);
    //TFitResultPtr r = hEPDRefMult[i]->Fit(f, "S");
    //TMatrixDSym cov = r->GetCovarianceMatrix();
    /*for (int j = 0; j < polyFit+1; ++j)
    {
      //cout.setf(ios::scientific);
      RefMultFile<<std::scientific;
      RefMultFile<<Form("%f \t",r->Value(j));
      RefMultFile<<endl;
    }*/
    c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf");
    hEPDRefMult[i]->Write();
    //RingFile[i].close();
  }

  c1->SaveAs("/mnt/d/27gev_production/data/EPDFit.pdf]");

  MyFile->Close();
  RefMultFile.close();
  picoReader->Finish();
  
}