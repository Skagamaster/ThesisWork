// ----------------------------------------------------------------------------------------------------
//
// this macro begins with RunSimulator.C but rather than using my TrivialEventGenerator, I use UrQMD
//
// ----------------------------------------------------------------------------------------------------

/*
 So, you've downloaded the StRoot/StPicoEvent and StRoot/StEpdUtil packages onto your laptop, eh?  Great!!
 Now you've got to do a few more things, to make the code useable
 
 1.  TWO quick edits on StRoot/StPicoEvent/StPicoEpdHit.cxx, right at the bottom:
 i)  comment out //#include "StPicoMessMgr.h"
 and replace with #include <iostream>
 ii) replace LOG_INFO << "EPD hit id: " << mId << " QT data: " << mQTdata << " nMIP: " << mnMIP << endm;
 with  std::cout << "EPD hit id: " << mId << " QT data: " << mQTdata << " nMIP: " << mnMIP << std::endl;
 
 2. a quick edit to StRoot/StEpdUtil/StEpdFastSim/StEpdFastSim.cxx at the very top.
 You will see an //#include statement to un-comment.  Basically, you need to point directly to StPicoEpdHit.h file
 
 3. Similar in StRoot/StEpdUtil/StEpdEpFinder.cxx
 
 4. go to StRoot/StPicoEvent
 root
 > .L StPicoEpdHit.cxx++
 > .q
 
 5. go to StRoot/StEpdUtil
 root
 > .L StEpdGeom.cxx++
 > .L StEpdEpInfo.cxx++
 > .L StEpdEpFinder.cxx++  <-- this one will give you lots of warnings about possibly-undefined variables.  Ignore them.
 > .q
 
 6. go to StRoot/StEpdUtil/StEpdFastSim
 root
 > .L ../StEpdGeom_cxx.so
 > .L StEpdTrivialEventGenerator.cxx++
 > .L StEpdFastSim.cxx++
 
 Now, steps 4-6 above make the .so files.  These files are loaded (see below) with R__LOAD_LIBRARY commands.
 That's how Root 6 does it.  In Root 5, they use the gSystem->Load() commands, and you'll just have to figure
 it out.  (Or, get with the times and install root6.)
 
 And, if you want to run it on RCF rather than at home, that's fine, too.  You'll just need to remove the
 R__LOAD_LIBRARY statements and maybe screw around a bit, but it shouldn't be too hard.
 
 - Mike Lisa lisa.1@osu.edu - 15feb2020
 */

R__LOAD_LIBRARY(./StEpdTrivialEventGenerator_cxx.so)
R__LOAD_LIBRARY(../StEpdGeom_cxx.so)
R__LOAD_LIBRARY(../StEpdEpInfo_cxx.so)
R__LOAD_LIBRARY(../StEpdEpFinder_cxx.so)
R__LOAD_LIBRARY(../../StPicoEvent/StPicoEpdHit_cxx.so)
R__LOAD_LIBRARY(./StEpdFastSim_cxx.so)

// ----------------------------------------------------------------------------------------------------
class UrQMDeg{
public:
    UrQMDeg();
    ~UrQMDeg(){};
    void AddFile(TString filename);
    TClonesArray* Momenta(int ievent=-1);  // if left at default of -1, then it simply gives event # mEventNumber and increments mEventNumber
    int NumEventsMax(){return mChain->GetEntries();}
    void SetBrange(double Blow, double Bhigh){mBrange[0]=Blow; mBrange[1]=Bhigh;}  // select impact parameters
    float B(){return mB;}
    int Mulp(){return mMulp;}
    int MulpFwd(){return mMulpFwd;}
private:
    TChain* mChain;
    TClonesArray* mTracks;
    int mEventNumber;
    float mB;
    int mMulp;
    int mMulpFwd;
    int mMul;
    int mPid[200000];
    float mgpx[200000];
    float mgpy[200000];
    float mgpz[200000];
    float mBrange[2];
};
// ----------------------------------------------------------------------------------------------------
UrQMDeg::UrQMDeg(){
    mEventNumber=0;
    mTracks = new TClonesArray("TVector3",5000);
    mChain = new TChain("h1");
    mChain->SetBranchAddress("parimp",&mB);
    mChain->SetBranchAddress("mpart",&mMul);
    mChain->SetBranchAddress("igid",&mPid);
    mChain->SetBranchAddress("gpx",&mgpx);
    mChain->SetBranchAddress("gpy",&mgpy);
    mChain->SetBranchAddress("gpz",&mgpz);

    mBrange[0] = -100.0;  mBrange[1] = 100.0;    // i.e. no cut

}
// ----------------------------------------------------------------------------------------------------
void UrQMDeg::AddFile(TString f){mChain->Add(f.Data());}

// ----------------------------------------------------------------------------------------------------
TClonesArray* UrQMDeg::Momenta(int ievent){
    //x = (condition) ? (value_if_true) : (value_if_false);
    //int evNum = (ievent<0)?mEventNumber++:ievent;
    int evNum = ievent; mEventNumber = ievent;
    
    if (evNum>=mChain->GetEntries()) return 0;
    mChain->GetEntry(evNum);
      //cout << "event# / b / mul : " << mEventNumber << " " << mB << " " << mMul << endl;
    if ((mB>mBrange[1])||(mB<mBrange[0])) return 0;  // outside of range
    mTracks->Clear();
    
    mMulp = 0; mMulpFwd = 0;
    int Nmidrap = 0;
   //cout<<"Total in momenta loop "<<mMul<<endl;
    
    for (int i=0; i<mMul; i++){
        // pid=0/pi+ 1/k+ 2/p 3/pi- 4/k- 5/pbar 6/pi0 7/eta 8/k0 9/n 10/nbar 11/default
        //Bill's pid numbers are geant
        //8/pi+ 9/pi- 11/k+ 12/k- 14/proton 15/pbar
        
        bool isgood = false;
        if ((mPid[i] == 8)||(mPid[i] == 9)||(mPid[i] == 11)||(mPid[i] == 12)||(mPid[i] == 14)||(mPid[i] == 15))
            isgood = true;
        if (! isgood)
            continue;
        
        //cout<<mPid[i]<<endl;
        //if (mPtbin[i]==0) continue;
        //cout<<"good track"<<endl;
        TVector3* v = (TVector3*)mTracks->ConstructedAt(mTracks->GetEntries());
        //v->SetPtEtaPhi((double)mPtbin[i]/100.0,(double)mEtabin[i]*0.02,(double)mPhibin[i]*TMath::Pi()/256.0);
        v->SetXYZ(mgpx[i],mgpy[i],mgpz[i]);
        //cout<<"pxpypz = "<<mgpx[i]<<" "<<mgpy[i]<<" "<<mgpz[i]<<endl;
        if ((abs(v->Eta())<1.0))
            Nmidrap++;
        if ((abs(v->Eta())<1.0)&&((mPid[i] == 2) || (mPid[i]==5)))
            mMulp++;
        if (((abs(v->Eta())<5.1) && (abs(v->Eta())>2.1))&&((mPid[i] == 2) || (mPid[i]==5)))
             mMulpFwd++;
         
    }
     
    //cout<<"mMulp is "<<mMulp<<" and mMulpFwd is "<<mMulpFwd<<" and miprap is "<<Nmidrap<<" mtracks total "<<mTracks->GetEntries()<<endl;
    //  cout << "Number of momenta I am returning is .... " << mTracks->GetEntries() << endl;
    return mTracks;
}
// ----------------------------------------------------------------------------------------------------


// --------------------------- Put in "bad detector efficiency region" --------------------------------
bool BadDetector(TVector3* mom){
    //  if ((mom->Eta()<1.2)&&(mom->Eta()>0.6)&&(mom->Phi()<-1.0)&&(mom->Phi()>-1.6)) return true;
    //  if ((mom->Eta()<-0.8)&&(mom->Eta()>-1.2)&&(mom->Phi()<2.7)&&(mom->Phi()>2.0)) return true;
    return false;
}
// ----------------------------------------------------------------------------------------------------

int ReturnEpdEta(TVector3* t){
    float Epdeta[] = {2.14,2.2,2.27,2.34,2.41,2.50,2.59,2.69,2.81,2.94,3.08,3.26,3.47,3.74,4.03,4.42,5.09};//epd eta values
    //backwards, damn
    int row = 0;
    if ((abs(t->Eta())<Epdeta[0]) || ((abs(t->Eta())>Epdeta[16])))
        return row; ///not in epd at all
    for (int i = 0;i<17;i++){
        if (abs(t->Eta()) < Epdeta[i]){
            row = 17-i; //because values are backwards
            i = 17;
        }
    }
    return row;
    
}

void RunSimulatorUrQMDMLRR_Test(int Nevents=500000,string outfile = "7"){
    
    StEpdGeom* eGeom = new StEpdGeom();    // handy later
    
    // Note what I am doing in the constructor here.  I am purposely making it such that there is no phi-weighting nor shifting weights.
    // This is exactly what we want for a UrQMD simulation because
    // 1) there is no noise or deadness or calibration to worry about, so it is unneeded
    // 2) we will be dealing with low statistics and using the same data over and over, so there will be statistical fluctuations in the flux which
    //    we do NOT want to divide out.  (If this is not immediately obvious, then please just think it through.)
    //
    // If I had just used the default values, then the same thing would happen, but here I just want to make it very clear that this is on purpose.
    StEpdEpFinder* epFinder = new  StEpdEpFinder(1,"IgnoreThisFile.root","ThisFileDoesntExist.root");
    epFinder->SetnMipThreshold(0.3);   // our standard
    epFinder->SetMaxTileWeight(2.0);   // our standard
    TH2D etaWeights("etaWeightUrQMD","etaWeightUrQMD",13,1.8,5.7,1,0.5,1.5);
    etaWeights.Fill(1.95,1,0.0274309);
    etaWeights.Fill(2.25,1,0.0342367);
    etaWeights.Fill(2.55,1,0.0381357);
    etaWeights.Fill(2.85,1,0.0254362);
    etaWeights.Fill(3.15,1,0.00278843);
    etaWeights.Fill(3.45,1,-0.0312159);
    etaWeights.Fill(3.75,1,-0.0556061);
    etaWeights.Fill(4.05,1,-0.0732146);
    etaWeights.Fill(4.35,1,-0.0734148);
    etaWeights.Fill(4.65,1,-0.0206689);
    etaWeights.Fill(4.95,1,-0.00567578);
    etaWeights.Fill(5.25,1,-0.00229185);
    etaWeights.Fill(5.55,1,-0.00261793);
    epFinder->SetEtaWeights(1,etaWeights);
    
    // ================================  Step 1  ==============================
    /* =================================================================
     Here, I will use the UrQMD files that Xiaoyou found for me
     =================================================================  */
    UrQMDeg* eg = new UrQMDeg();
    eg->AddFile("/mnt/d/newfile/7/RACF/out_7.root");
    /*
    for (int i=253; i<670; i++){
        //for (int i=255; i<256; i++){
        //eg->AddFile(Form("/Volumes/Rosi\ Back-Up/UrQMD032020/200GeV/out%03i.root",i));
        //eg->AddFile(Form("/Volumes/Rosi\ Back-Up/UrQMD032020/19_6GeV/out%03i.root",i));
        //eg->AddFile(Form("/Users/rosireed/Desktop/EPD/UrQMDtest/urqmd007/urqmd_19_0099_02013%i.root",i));
        //eg->AddFile(Form("/Users/rosireed/Desktop/EPD/UrQMDtest/urqmd011/urqmd_20_0099_02013%i.root",i));
        //eg->AddFile(Form("/Users/rosireed/Desktop/EPD/UrQMDtest/urqmd015/urqmd_31_0099_02013%i.root",i));
        eg->AddFile(Form("/mnt/d/newfile/19/RACF/urqmd_23_0099_02013%i.root",i));
    }
    */
    //  eg->SetBrange(3.0,10.0);
    cout << " Okay, the UrQMD event generarator can give us " << eg->NumEventsMax() << " events\n";
    
    // ================================  Step 2  ===============================
    /* Loop over events
     In this process, the parts are:
     a.  Generate an event with an event generator (Trivial, UrQMD, whatever).  Format is a TClonesArray of momentum TVector3 objects
     b.  If you want the events randomly rotated, or whatever, do it yourself.  That way you control stuff and you have the event plane angle.
     c.  Hand this TClonesArray(TVector3) to the StEpdFastSimulator, which will produce a TClonesArray of StPicoEpdHit objects, just like the real data.
     In this step, you have to tell the simulator where is the primary vertex.  You, the user, control this.  Again, this way you know, and you can do analysis or whatever.
     d.  Now do your regular EPD analysis!  You can hand this TClonesArray(StPicoHit) to the StEpdEpFinder or whatever you want
     
     As usual, the code below is longer than strictly necessary, since I put in comments and fill histograms, etc.
     Right here is what the code looks like, pared down:
     -----
     TRandom3* ran = new TRandom3;
     int Nevents=500;
     StEpdFastSim* efs = new StEpdFastSim(0.2);
     for (int iev=0; iev<Nevents; iev++){
     TClonesArray* momenta = eg->Momenta();
     double RPangle = ran->Uniform(2.0*TMath::Pi());
     for (int trk=0; trk<momenta->GetEntries(); trk++){
     TVector3* mom = (TVector3*)momenta->At(trk);
     mom->RotateZ(RPangle);}
     TVector3 PrimaryVertex(0.0,0.0,0.0);
     TClonesArray* picoHits = efs->GetPicoHits(momenta,PrimaryVertex);
     // now you do something with this data
     }
     -----
     */
    
    
    // the following histograms are just standard for looking at stuff.  In principle they are optional
    TH2D* EtaPhi = new TH2D("EtaPhi","Eta-Phi from generator",50,-8,8,50,-TMath::Pi(),TMath::Pi());
    TH1D* dNdEta = new TH1D("dNdEta","dNdEta",50,-8,8);
    TH1D* nmip   = new TH1D("Nmip","Nmip of all tiles",50,0.0,10.0);
    TH2D* EastWheel = new TH2D("East","East EPD",100,-100.0,100.0,100,-100.0,100.0);
    TH2D* WestWheel = new TH2D("West","West EPD",100,-100.0,100.0,100,-100.0,100.0);
    TH2D* EastWheelADC = new TH2D("EastADC","East EPD - ADC weighted",100,-100.0,100.0,100,-100.0,100.0);
    TH2D* WestWheelADC = new TH2D("WestADC","West EPD - ADC weighted",100,-100.0,100.0,100,-100.0,100.0);
    
    TH2D* Psi1eastVsPsi1west = new TH2D("Psi1eastVsPsi1west","#Psi_{1}^{East} vs #Psi_{1}^{West}",30,0.0,2.0*TMath::Pi(),30,0.0,2.0*TMath::Pi());
    TH2D* Psi2eastVsPsi2west = new TH2D("Psi2eastVsPsi2west","#Psi_{2}^{East} vs #Psi_{2}^{West}",30,0.0,TMath::Pi(),30,0.0,TMath::Pi());;
    
    TH2D* SkipperCent = new TH2D("Mock","Mock quantity versus impact parameter",40,0,15,40,0,300);
    
    TProfile* v1VersusFloor = new TProfile("v1VersusFloor","v1 using #Psi=0",40,-6,6);
    TProfile* v1VersusRP1 = new TProfile("v1VersusEP1EPD","v1 using #Psi_{1,EPD}",40,-6,6);
    
    TH1D* dNdB = new TH1D("dNdB","dNdB",100,0.0,18.0);
    
    // end of optional histograms
    
    TRandom3* ran = new TRandom3;
    if (Nevents<0) Nevents = eg->NumEventsMax();
    
    StEpdFastSim* efs = new StEpdFastSim(0.2);    // the argument is WID/MPV for the Landau energy loss.  Use 0.2 for the EPD
    
    TFile* ntupFile = new TFile(Form("/mnt/d/UrQMD_cent_sim/%s/CentralityNtuple.root",outfile.c_str()),"RECREATE");
    TNtuple* centNT = new TNtuple("Rings","Ring contents and b and N_TPC",
                                  "r01:r02:r03:r04:r05:r06:r07:r08:r09:r10:r11:r12:r13:r14:r15:r16:TpcMult:b:ParticleMult:RefMult1:RefMult2:RefMult3:FwdAll:Fwd1:Fwd2:Fwd3:FwdAllp:NmipRaw:r01part:r02part:r03part:r04part:r05part:r06part:r07part:r08part:r09part:r10part:r11part:r12part:r13part:r14part:r15part:r16part");
    float values[44];    // the first 16 of these are the ring contents; values[16] is TPC multiplicity; values[17] is impact parameter (b)
    TNtuple* refMults = new TNtuple("refmults","It's the refmults","b:RefMult1:RefMult2:RefMult3:Fwd1:Fwd2:Fwd3:FwdAll:FwdAllp");
    float vals[9];
    
    Nevents = eg->NumEventsMax();
    //Nevents = 1000;
    for (int iev=0; iev<Nevents; iev++){
        if (iev%1000==0) cout << "On event " << iev << " / " << Nevents << endl;
        
        for (int j=0; j<44; j++){values[j]=0.0;}
        
        // a.  Generate event
        TClonesArray* momenta = eg->Momenta(iev);
        if (momenta==0){
            //      cout << "No event!!\n";
            continue;
        }
        // b.  Randomly rotate (optional)
        double RPangle = ran->Uniform(2.0*TMath::Pi());
        for (int trk=0; trk<momenta->GetEntries(); trk++){
            TVector3* mom = (TVector3*)momenta->At(trk);
            mom->RotateZ(RPangle);
        }
        
        int RefMult1 = 0; int RefMult2 = 0; int RefMult3 = 0;
        int Fwd1 = 0; int Fwd2 = 0; int Fwd3 = 0;
        int FwdAll = 0; int FwdAllp = 0;
        
        dNdB->Fill(eg->B());
        
        for (int itrk=0; itrk<momenta->GetEntries(); itrk++){
            TVector3* mom = (TVector3*)momenta->At(itrk);
            if (abs(mom->Eta())<0.5)
                RefMult1++;
            if ((abs(mom->Eta())<1.0) && (abs(mom->Eta())>0.5))
                RefMult2++;
            if (abs(mom->Eta())<1.0)
                RefMult3++;
            if ((abs(mom->Eta()) < 5.1) && (abs(mom->Eta())>2.1))
                FwdAll++;
            if ((abs(mom->Eta()) < 3.0) && (abs(mom->Eta())>2.1))
                Fwd1++;
            if ((abs(mom->Eta()) < 4.0) && (abs(mom->Eta())>3.0))
                Fwd2++;
            if ((abs(mom->Eta()) < 5.0) && (abs(mom->Eta())>4.0))
                Fwd3++;
            
            int row = ReturnEpdEta(mom);
            if (row > 0)
                values[27+row]+=1.0; //adding up the ring weights for particles
            
            if (BadDetector(mom)) continue;
            EtaPhi->Fill(mom->Eta(),mom->Phi());
            dNdEta->Fill(mom->Eta());
            v1VersusFloor->Fill(mom->Eta(),cos(mom->Phi()));
        } // just a histogram
        
        // c.  Run EPD Fast simulator
        TVector3 PrimaryVertex(0.0,0.0,0.0);
        TClonesArray* picoHits = efs->GetPicoHits(momenta,PrimaryVertex);  // and that's it!  you've got the TClonesArray of StPicoHit objects
        
        if (picoHits->GetEntries()==0) continue;
        
        StEpdEpInfo EPresults = epFinder->Results(picoHits,PrimaryVertex,0);
        Psi1eastVsPsi1west->Fill(EPresults.EastRawPsi(1),EPresults.WestRawPsi(1));  // Use RAW in this case!
        Psi2eastVsPsi2west->Fill(EPresults.EastRawPsi(2),EPresults.WestRawPsi(2));  // Use RAW in this case!
        
        //    double AngleToUseForV1 = RPangle;                  // "perfect"
        double AngleToUseForV1 = EPresults.FullRawPsi(1);  // "real"
        
        
        /* fill some diagnostic plots */
        for (int i=0; i<picoHits->GetEntries(); i++){        // quick plots
            StPicoEpdHit* ph = (StPicoEpdHit*)picoHits->At(i);
            nmip->Fill(ph->nMIP());
            TVector3 point = eGeom->RandomPointOnTile(ph->id());
            if (ph->id()<0){ // East
                EastWheel->Fill(point.X(),point.Y());
                EastWheelADC->Fill(point.X(),point.Y(),ph->nMIP());
            }
            else{             // West
                WestWheel->Fill(point.X(),point.Y());
                WestWheelADC->Fill(point.X(),point.Y(),ph->nMIP());
            }
            TVector3 line = point-PrimaryVertex;
            v1VersusRP1->Fill(line.Eta(),cos(line.Phi()-AngleToUseForV1));
        }
        
        // now v1 only for TPC tracks
        for (int itrk=0; itrk<momenta->GetEntries(); itrk++){
            TVector3* mom = (TVector3*)momenta->At(itrk);
            if (abs(mom->Eta())>1.8) continue;
            if (BadDetector(mom)) continue;
            //EtaPhi->Fill(mom->Eta(),mom->Phi());
            //dNdEta->Fill(mom->Eta());
            v1VersusRP1->Fill(mom->Eta(),cos(mom->Phi()-AngleToUseForV1));
            if (abs(mom->Eta())<1.0) values[16]+=1.0;
        }
    
        float NmipRaw = 0;
        
        // and, for a Skipper-type analysis....
        // I will plot 2*Ring5 + 1.5*Ring2, weighted by truncated nMip [truncated at 2]
        double mockCent(0.0);
        for (int i=0; i<picoHits->GetEntries(); i++){
            StPicoEpdHit* ph = (StPicoEpdHit*)picoHits->At(i);
            int row = ph->row();                               // 'row' is the same as 'ring'
            float w = ph->nMIP();
            NmipRaw+=w;
            if ((row > 1) && (w>2.0))
                w=2.0;
            else if ((row == 1)&&(w>3.0))
                w=3.0;
            values[row-1] += w;
            if (row==5) mockCent += 2.0*w;
            else if (row==2) mockCent += 1.5*w;
        }
        SkipperCent->Fill(eg->B(),mockCent);
        
        values[17] = eg->B();
        values[18] = momenta->GetEntries();
        values[19] = RefMult1;
        values[20] = RefMult2;
        values[21] = RefMult3 - eg->Mulp();
        values[22] = FwdAll;
        values[23] = Fwd1;
        values[24] = Fwd2;
        values[25] = Fwd3;
        values[26] = FwdAll - eg->MulpFwd();
        values[27] = NmipRaw;
       //27 beyond already filled
        vals[0] = eg->B();
        vals[1] = RefMult1;
        vals[2] = RefMult2;
        vals[3] = RefMult3 - eg->Mulp();
        vals[4] = Fwd1;
        vals[5] = Fwd2;
        vals[6] = Fwd3;
        vals[7] = FwdAll;
        vals[8] = FwdAll - eg->MulpFwd();
        refMults->Fill(vals);

        centNT->Fill(values);
      //  cout<<endl;
    }
    
    dNdEta->Scale(1.0/(dNdEta->GetXaxis()->GetBinWidth(2)*(double)dNdB->GetEntries()));
    
    dNdEta->Write();
    ntupFile->Write();
    ntupFile->Close();
    
    
    TCanvas* tc = new TCanvas("diagnostics","diag",1600,1200);
    tc->Divide(4,3);
    tc->cd(1);  EtaPhi->Draw("colz");
    tc->cd(2); nmip->Draw();
    tc->cd(5)->SetLogz(); EastWheel->Draw("colz");
    tc->cd(6)->SetLogz(); WestWheel->Draw("colz");
    tc->cd(9)->SetLogz(); EastWheelADC->Draw("colz");
    tc->cd(10)->SetLogz(); WestWheelADC->Draw("colz");
    tc->cd(3); dNdEta->Draw();
    tc->cd(4); v1VersusFloor->Draw();
    tc->cd(11); Psi1eastVsPsi1west->Draw("colz");
    tc->cd(12); Psi2eastVsPsi2west->Draw("colz");
    
    tc->cd(7); SkipperCent->Draw("colz");
    
    tc->cd(8); v1VersusRP1->Draw();
    
}


