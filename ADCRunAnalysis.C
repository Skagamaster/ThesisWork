/*
__________________________________________
  Original code by Grigory.
  Updated code by Prashanth.
  Tweaks by Skipper.
------------------------------------------
*/


/// C++ headers
#include <iostream>

/// PicoDst headers
#include "StPicoDstReader.h"
#include "StPicoDst.h"
#include "StPicoEvent.h"
#include "StPicoTrack.h"
#include "StPicoEpdHit.h"

/// ROOT headers
#include "TFile.h"
#include "TChain.h"
#include "TSystem.h"
#include "TH1.h"
#include "TMath.h"


R__LOAD_LIBRARY(libStPicoDst)

//_________________
void ADCRunAnalysis(const Char_t *inFile = 
"/mnt/d/14GeV/Picos/st_physics_adc_20097008_raw_1000003.picoDst.root") {


  // Array for ADC values
  double data[4464];
  double ADCs[2][12][31];

  // Finding the day parameter.
  std::string Queso = inFile;
  Queso = Queso.substr( Queso.find_last_of('/')+1 );
  std::string Nachos = Queso.substr(17,3);
  int Tacos = stoi(Nachos);

/// File for normalising the nMips
std::ifstream input(Form("/mnt/d/14Gev/NmipConstantsDay%d.txt",Tacos));
    for (int i = 0; i < 4464; ++i)
    {
      input >> data[i];
    }
    for (int i = 0; i < 2; ++i)
    {
      for (int j = 0; j < 12; ++j)
      {
        for (int k = 0; k < 31; ++k)
        {
          ADCs[i][j][k] = data[(i*2232)+(j*186)+k*6+4];
        }
      }
    }

  std::cout << "Feed me data!" << std::endl;

  cout <<" analysis file "<<inFile<<endl;
  StPicoDstReader* picoReader = new StPicoDstReader(inFile);
  picoReader->Init();

  /// This is a way if you want to spead up IO
  std::cout << "Explicit read status for some branches" << std::endl;
  picoReader->SetStatus("*",0);
  picoReader->SetStatus("Event",1);
  picoReader->SetStatus("Track",1);
  picoReader->SetStatus("EpdHit",1);
  std::cout << "Statuses set. M04r data!" << std::endl;

  std::cout << "Beginning to chew . . ." << std::endl;

  Long64_t events2read = picoReader->chain()->GetEntriesFast();
  //Long64_t events2read = 10;

  std::cout << "Number of events to read: " << events2read << std::endl;

  /// Histogramming

  // 1D histograms for ADC distributions
  TH1D *mAdcDists[2][12][31];
  // 1D histograms for nMip distributions
  TH1D *mNmipDists[2][12][31];

  for (int ew=0; ew<2; ew++){
    for (int pp=1; pp<13; pp++){
      for (int tt=1; tt<32; tt++){
    mAdcDists[ew][pp-1][tt-1]  = new TH1D(Form("AdcEW%dPP%dTT%d",ew,pp,tt),
      Form("AdcEW%dPP%dTT%d",ew,pp,tt),4096,0,4096);
    mNmipDists[ew][pp-1][tt-1] = new TH1D(Form("nMipEW%dPP%dTT%d",ew,pp,tt),
      Form("nMipEW%dPP%dTT%d",ew,pp,tt),4096,0,40);
      }
    }
  }
  /*
  TH1F *hRefMult = new TH1F("hRefMult","Reference multiplicity;refMult", 500, -0.5, 499.5);
  TH1F *hTransvMomentum = new TH1F("hTransvMomentum", 
                                    "Track transverse momentum;p_{T} (GeV/c)", 200, 0., 2.);
  */

  /// Event loop
  for(Long64_t iEvent=0; iEvent<events2read; iEvent++) {

    //std::cout << "Obey your orders, Master! Working on event #["
    //         << (iEvent+1) << "/" << events2read << "]" << std::endl;

    Bool_t readEvent = picoReader->ReadPicoEvent(iEvent);
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
    //hRefMult->Fill( event->refMult() );

    /// Track analysis
    
    /*Int_t nTracks = dst->numberOfTracks();
    std::cout << "Number of tracks in event: " << nTracks << std::endl;
    /// Loop over tracks (No need for this application. SK)
    for(Int_t iTrk=0; iTrk<nTracks; iTrk++) {

      StPicoTrack *picoTrack = dst->track(iTrk);
      if(!picoTrack) continue;
      //std::cout << "Track #[" << (iTrk+1) << "/" << nTracks << "]"  << std::endl;

      /// Single-track cut example
      if( !picoTrack->isPrimary() ||
	  picoTrack->nHits() < 15 ||
	  TMath::Abs( picoTrack->gMom().pseudoRapidity() ) > 0.5 ) {
	continue;
      } //for(Int_t iTrk=0; iTrk<nTracks; iTrk++)

      hTransvMomentum->Fill( picoTrack->gPt() );
    }*/ 

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
      mNmipDists[ew][pp-1][tt-1]->Fill(adc/ADCs[ew][pp-1][tt-1]);

    }


  } //for(Long64_t iEvent=0; iEvent<events2read; iEvent++)

  picoReader->Finish();

  /*std::string Queso = inFile;
  Queso = Queso.substr( Queso.find_last_of('/')+1 );
  std::string Nachos = Queso.substr(13,3);
  int Tacos = stoi(Nachos);*/
  if (Tacos < 100)
  {
    TString pathSave = Form("/mnt/d/14GeV/Day%d/",Tacos);
    TFile *MyFile = TFile::Open(pathSave+Queso,"RECREATE");
    for (int ew=0; ew<2; ew++){
    for (int pp=1; pp<13; pp++){
      for (int tt=1; tt<32; tt++){
    mAdcDists[ew][pp-1][tt-1]->Write();
    mNmipDists[ew][pp-1][tt-1]->Write();
      }
    }
  }

  MyFile->Close();

  }
  else
  {
    TString pathSave = Form("/mnt/d/14GeV/Day%d/",Tacos);
    TFile *MyFile = TFile::Open(pathSave+Queso,"RECREATE");
    for (int ew=0; ew<2; ew++){
    for (int pp=1; pp<13; pp++){
      for (int tt=1; tt<32; tt++){
    mAdcDists[ew][pp-1][tt-1]->Write();
    mNmipDists[ew][pp-1][tt-1]->Write();
      }
    }
  }

  MyFile->Close();

  }

  std::cout << "<burp> Ahhhh. All done; thanks!" << std::endl;
}
