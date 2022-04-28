#define _VANILLA_ROOT_
// C++ headers
#include <iostream>
#include <sstream>

// ROOT headers
#include "TROOT.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TFile.h"
#include "THStack.h"
#include "TLegend.h"
#include "TChain.h"
#include "TTree.h"
#include "TSystem.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
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

bool isGoodEvent(StPicoEvent* event,double nTracks);
bool isGoodTrack(StPicoTrack* track, TVector3 pVtx);
int pBin(double pT);

/// inFile - is a name of name.picoDst.root file or a name
///          of a name.lis(t) files that contains a list of
///          name1.picoDst.root files

void SkandaWeightCalculations(const Char_t *inFile = "/mnt/d/27gev_production/st_physics_19999_raw_1234.picoDst.root") { //include the name of the file i want to run on

    // Loop over events again to recalculate event plane
    std::cout << "Hi! Lets do some physics, mister!" << std::endl;

    StPicoDstReader* picoReader = new StPicoDstReader(inFile);
    picoReader->Init();

    // This is a way if you want to spead up IO
    std::cout << "Explicit read status for some branches" << std::endl;
    picoReader->SetStatus("*",0);
    picoReader->SetStatus("Event",1);
    picoReader->SetStatus("Track",1);
    picoReader->SetStatus("EpdHit",1);
    picoReader->SetStatus("BTofHit",1);
    picoReader->SetStatus("BTofPidTraits",1);
    std::cout << "Status has been set" << std::endl;

    std::cout << "Now I know what to read, mister!" << std::endl;

    if( !picoReader->chain() ) {
        std::cout << "No chain has been found." << std::endl;
    }
    Long64_t eventsInTree = picoReader->tree()->GetEntries();
    std::cout << "eventsInTree: "  << eventsInTree << std::endl;
    Long64_t events2read = picoReader->chain()->GetEntries();

    std::cout << "Number of events to read: " << events2read
    << std::endl;

    //__________________________________________________________________
    // Histogramming

    // Phi weight histograms - array split by east/west, positive/negative,
    //                         and bins of momentum:

    TH1* hTPCPhi[2][2][5];
    TH1* hTPCPhiWeight[2][2][5];

    for (int ew = 0; ew < 2; ew++) {
        for (int ch = 0; ch < 2; ch++) {
            for (int pTBin = 0; pTBin < 5; pTBin++) {
                hTPCPhi[ew][ch][pTBin] = new TH1F(Form("hTPCPhi_%d_%d_%d",ew,ch,pTBin),
                    Form("Phi distribution, ew=%d, ch=%d, pTBin=%d",ew,ch,pTBin),
                    630,-3.15,3.15);
                hTPCPhiWeight[ew][ch][pTBin] = new TH1F(Form("hTPCPhiWeight_%d_%d_%d",ew,ch,pTBin),
                    Form("Phi weights, ew=%d, ch=%d, pTBin=%d",ew,ch,pTBin),
                    630,-3.15,3.15);
            }
        }
    }

    //_______________________________________________________________
    // Loop over events
    for(Long64_t iEvent=0; iEvent<events2read; iEvent++) {  // change events2read to 200 to decrease runtime when testing

        if (iEvent % 10000 == 0) std::cout << "Working on event #[" << (iEvent+1) << "/" << events2read << "]" << std::endl;

        Bool_t readEvent = picoReader->readPicoEvent(iEvent);
        if( !readEvent ) {
            std::cout << "Something went wrong, mister! Nothing to analyze..."
            << std::endl;
            break;
        }

        // Retrieve picoDst
        StPicoDst *dst = picoReader->picoDst();

        // Retrieve event information
        StPicoEvent *event = dst->event();
        if( !event ) {
            std::cout << "Something went wrong, mister! Event is hiding from me..."
            << std::endl;
            break;
        }

        Int_t nTracks = dst->numberOfTracks();
        TVector3 pVtx = event->primaryVertex();

        // Event cuts
        if( !isGoodEvent(event,nTracks) ) continue;                     // Simple event cut
        if( pVtx.Z() < -40 || pVtx.Z() > 40 ) continue;                 // Condition that |V_z| < 40 cm
        if( pVtx.X()*pVtx.X() + pVtx.Y()*pVtx.Y() > 4 ) continue;       // Condition that |V_r| < 2 cm
        if( TMath::Abs(pVtx.Z() - event->vzVpd()) > 2 ) continue;       // Condition that |V_z,TPC - V_z,VPD|< 2 cm
        if (dst->numberOfEpdHits() < 10) continue;                      // Want some sort of signal in the EPD

        // TPC track analysis

        // Track loop
        for(Int_t iTrk=0; iTrk<nTracks; iTrk++) {

            // Retrieve i-th pico track
            StPicoTrack *picoTrack = dst->track(iTrk);

            // Track cuts
            if(!picoTrack) continue;
            if( !isGoodTrack(picoTrack, pVtx) ) continue;        // Simple single-track cut
            if ( !(picoTrack->isPrimary()) ) continue;

            double pT = picoTrack->pMom().Pt();
            double eta = picoTrack->pMom().Eta();
            double phi = picoTrack->pMom().Phi();

            // Identify bins
            int EW = ( eta < 0 )?0:1;                   // ew: east/west
            int CH = ( picoTrack->charge() < 0 )?0:1;   // ch: negative/positive
            int PtBin = pBin(pT)/2;

            hTPCPhi[EW][CH][PtBin]->Fill( phi );

            TH1* h;
            int bin;
            double w;

            h = hTPCPhiWeight[EW][CH][PtBin];

            bin = h->FindBin(phi);
            w = ( h->GetBinContent(bin) )/( h->GetMean() );

        } //for(Int_t iTrk=0; iTrk<nTracks; iTrk++)

    } //for(Long64_t iEvent=0; iEvent<events2read; iEvent++)

    std::cout << "I'm done with analysis. We'll have a Nobel Prize, mister!"
    << std::endl;

    // Writing to event plane output
    TFile fOutput("SkandaFlowCalculationsTPCCorrectionHistograms_INPUT.root","recreate");

    TH1* h;
    double w;

    for (int ew = 0; ew < 2; ew++) {
        for (int ch = 0; ch < 2; ch++) {
            for (int pTBin = 0; pTBin < 5; pTBin++) {
                hTPCPhi[ew][ch][pTBin]->Write();
                double NTotal = hTPCPhi[ew][ch][pTBin]->GetEntries();

                for (int bin = 1; bin <= 630; bin++) {
                    double NPhi = hTPCPhi[ew][ch][pTBin]->GetBinContent(bin);
                    hTPCPhiWeight[ew][ch][pTBin]->SetBinContent(bin, NTotal*((NPhi==0)?0:1/NPhi)/630. );
                }

                hTPCPhiWeight[ew][ch][pTBin]->Write();
            }
        }
    }

    fOutput.Close();

    picoReader->Finish();

}



bool isGoodEvent(StPicoEvent* event, double nTracks) {
  if ( nTracks < 10 ) return false;
  return true;
}



bool isGoodTrack(StPicoTrack* track, TVector3 pVtx) {
  if ( track->gMom().Mag() < 0.2 || track->gDCA(pVtx).Mag()>3. ) return false; // Eliminates tracks with bad resolution or that may be falsely associated
  if ( track->gMom().Mag() > 2 ) return false; // Eliminates tracks coming from high-pT (jet) particles
  double nHits = track->nHits();
  double nHitsPoss = track->nHitsPoss();
  if ( nHits/nHitsPoss < 0.52 ) return false;
  if ( nHits < 15 ) return false;
  return true;
}



int pBin(double pT) {
    if ( pT < 0.24 ) return 0;
    else if ( pT < 0.29 ) return 1;
    else if ( pT < 0.32 ) return 2;
    else if ( pT < 0.37 ) return 3;
    else if ( pT < 0.42 ) return 4;
    else if ( pT < 0.49 ) return 5;
    else if ( pT < 0.59 ) return 6;
    else if ( pT < 0.76 ) return 7;
    else if ( pT < 0.94 ) return 8;
    else return 9;
}