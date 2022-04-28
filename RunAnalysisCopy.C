/*
__________________________________________
  Original code by Grigory.
  Updated code by Prashanth.
  Tweaks by Skipper.
------------------------------------------
*/

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
#include "StPicoTrackCovMatrix.h"

// Load libraries (for ROOT_VERSTION_CODE >= 393215)
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,0,0) 
R__LOAD_LIBRARY(libStPicoDst)
#endif

//_________________
void RunAnalysisCopy(int the_run = 21250010) {

const Char_t *inFile = Form("/mnt/c/PhysicsProcessing/7.7GeV/250/%d/%d.picoDst.root", the_run, the_run);
//------------------------------------------------------------
  std::cout << "Feed me data!" << std::endl;
  cout <<" analysis file "<<inFile<<endl;
  StPicoDstReader* picoReader = new StPicoDstReader(inFile);
  picoReader->Init();

  /// This is a way if you want to spead up IO
  std::cout << "Explicit read status for some branches" << std::endl;
  picoReader->SetStatus("*",0);
  picoReader->SetStatus("Event",1);
  //picoReader->SetStatus("Track",1);
  picoReader->SetStatus("EpdHit",1);
  std::cout << "Statuses set. M04r data!" << std::endl;

  std::cout << "Beginning to chew . . ." << std::endl;

  Long64_t events2read = picoReader->chain()->GetEntriesFast();
  //Long64_t events2read = 10;

  std::cout << "Number of events to read: " << events2read << std::endl;

  /// Histogramming

  /// 1D histograms for ADC distributions
  TH1D *mAdcDists[2][12][31];

  for (int ew=0; ew<2; ew++){
    for (int pp=1; pp<13; pp++){
      for (int tt=1; tt<32; tt++){
    mAdcDists[ew][pp-1][tt-1]  = new TH1D(Form("AdcEW%dPP%dTT%d",ew,pp,tt),
      Form("AdcEW%dPP%dTT%d",ew,pp,tt),4096,0,4096);
      }
    }
  }

  /// Event loop
  for(Long64_t iEvent=0; iEvent<events2read; iEvent++) {

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
      std::cout << "Event #" << iEvent << " does not exist. FEED ME!!" << std::endl;
      break;
    }

    if (TMath::Abs( sqrt(pow(event->primaryVertex().x(),2) + 
      pow(event->primaryVertex().y(),2))) > 2.0 )
    {
      continue;
    }
 

    // EPD Hit loop

    TClonesArray *mEpdHits = dst->picoArray(8); 
    //cout << mEpdHits->GetEntries()<<endl;
    StPicoEpdHit* epdHit;

    for (int hit=0; hit<mEpdHits->GetEntries(); hit++){
      epdHit = (StPicoEpdHit*)((*mEpdHits)[hit]);
      int ew = (epdHit->id()<0)?0:1;
      int pp = epdHit->position();
      int tt = epdHit->tile();
      double adc = epdHit->adc();
      mAdcDists[ew][pp-1][tt-1]->Fill(adc);
    }


  }
  picoReader->Finish();
  
//----------------------------------------------------


  std::cout << "<burp> Ahhhh. All done; thanks!" << std::endl;
}
