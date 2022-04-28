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

// inFile - is a name of name.picoDst.root file or a name
//          of a name.lis(t) files that contains a list of
//          name1.picoDst.root files

//_________________
void EPDADC(const Char_t *inFile = 
"/mnt/d/27gev_production/st_physics_19999_raw_12345.picoDst.root") {

	StPicoDstReader* picoReader = new StPicoDstReader(inFile);
	picoReader->Init();

	picoReader->SetStatus("*",0);
 	picoReader->SetStatus("Event",1);
  picoReader->SetStatus("Track",1);
  picoReader->SetStatus("EpdHit",1);

  Long64_t events2read = picoReader->chain()->GetEntriesFast();

  std::cout << "Number of events to read: " << events2read << std::endl;

  std::ofstream RefMultFile("/mnt/d/27gev_production/data/ADCGuys.txt",
    ofstream::out);

/// Event loop
  //for(Long64_t iEvent=0; iEvent<events2read; iEvent++) {
  for(Long64_t iEvent=0; iEvent<100; iEvent++) {

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

    RefMultFile << Form("Event # %d",iEvent) << endl;

    /// EPD Hit loop

    TClonesArray *mEpdHits = dst->picoArray(8); 
    StPicoEpdHit* epdHit;
    RefMultFile << Form("# of hits: %d",mEpdHits->GetEntries()) << endl << "[";

    for (int hit=0; hit<mEpdHits->GetEntries(); hit++){
      epdHit = (StPicoEpdHit*)((*mEpdHits)[hit]);
      RefMultFile << Form("%d \t",epdHit->adc());

      }

    RefMultFile << "]" << endl;
        
    }

  RefMultFile.close();
  picoReader->Finish();
  
}