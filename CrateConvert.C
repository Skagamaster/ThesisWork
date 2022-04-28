#include <fstream>      // std::filebuf

void CrateConvert(const Char_t *inFile = "/mnt/d/19GeV/19.6gev_new/20057/epdhist.20057003.1.root"){

// Finding the day parameter.
std::string Queso = inFile;
std::string Nachos = "Oh no!";
Queso = Queso.substr( Queso.find_last_of('/')+1 );
//Nachos = Queso.substr(0,2);
Nachos = Queso.substr(8,7);
std::cout << Nachos << std::endl;
int Tacos = stoi(Nachos);
std::cout << Tacos << std::endl;
TString pathSave = "/mnt/d/19GeV/";
TFile* MyFile = TFile::Open(pathSave+Form("Test%d.root",Tacos),"RECREATE");

TFile* in = new TFile(inFile);

    int ew = 0;
    int TT = 0;
    int PP = 0;
    for (int cr=1; cr<4; cr++){
    for (int bd=0; bd<15; bd++){
/// Ignores for bds not in use.
      if (bd==2)
      {
        continue;
      }
      if (bd==5)
      {
        continue;
      }
      if (bd==8)
      {
        continue;
      }
      if (bd==11)
      {
        continue;
      }
      if ((cr==3) && (bd==14))
      {
        continue;
      }

        for (int ch=0; ch<32; ch++){
          int ew = 0;

  /// Conversion from CMAC electronics to EW/PP/TT format.

          if ((cr==1) && (bd==0))
          {
            ew = 1;
            if ((ch>=0) && (ch<8))
            {
              TT = ch+24;
              PP = 1;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+16;
              PP = 2;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch+8;
              PP = 3;
            }
            else if (ch>23)
            {
              TT = ch;
              PP = 4;
            }
          }

          if ((cr==1) && (bd==1))
          {
            ew = 1;
            if ((ch>=0) && (ch<8))
            {
              TT = ch+24;
              PP = 5;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+16;
              PP = 6;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch+8;
              PP = 7;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch;
              PP = 8;
            }
          }

          if ((cr==1) && (bd==3))
          {
            ew = 1;
             if ((ch>=0) && (ch<8))
            {
              TT = ch+24;
              PP = 9;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+16;
              PP = 10;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch+8;
              PP = 11;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch;
              PP = 12;
            }
          }

          if ((cr==1) && (bd==4))
          {
            ew = 1;
            if ((ch>=0) && (ch<8))
            {
              TT = ch+16;
              PP = 1;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+8;
              PP = 2;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch;
              PP = 3;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch-8;
              PP = 4;
            }
          }

          if ((cr==1) && (bd==6))
          {
            ew = 1;
            if ((ch>=0) && (ch<8))
            {
              TT = ch+16;
              PP = 5;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+8;
              PP = 6;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch;
              PP = 7;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch-8;
              PP = 8;
            }
          }

          if ((cr==1) && (bd==7))
          {
            ew = 1;
            if ((ch>=0) && (ch<8))
            {
              TT = ch+16;
              PP = 9;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+8;
              PP = 10;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch;
              PP = 11;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch-8;
              PP = 12;
            }
          }
          
          if ((cr==1) && (bd==9))
          {
            ew = 1;
            if ((ch>=0) && (ch<4))
            {
              TT = ch+6;
              PP = 1;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-2;
              PP = 2;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-10;
              PP = 3;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-18;
              PP = 4;
            }
            else
            {
              continue;
            }
          }

          if ((cr==1) && (bd==10))
          {
            ew = 1;
            if ((ch>-1) && (ch<4))
            {
              TT = ch+6;
              PP = 5;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-2;
              PP = 6;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-10;
              PP = 7;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-18;
              PP = 8;
            }
            else
            {
              continue;
            }
          }

          if ((cr==1) && (bd==12))
          {
            ew = 1;
            if ((ch>=0) && (ch<4))
            {
              TT = ch+6;
              PP = 9;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-2;
              PP = 10;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-10;
              PP = 11;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-18;
              PP = 12;
            }
            else
            {
              continue;
            }
          }

          if ((cr==1) && (bd==13))
          {
            ew = 1;
            if ((ch>-1) && (ch<4))
            {
              TT = ch+2;
              PP = 1;
            }           
            else if ((ch>7) && (ch<12))
            {
              TT = ch-6;
              PP = 2;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-14;
              PP = 3;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-22;
              PP = 4;
            }
            else
            {
              continue;
            }
          }

          if ((cr==1) && (bd==14))
          {
            ew = 1;
            if ((ch>=0) && (ch<4))
            {
              TT = ch+2;
              PP = 5;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-6;
              PP = 6;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-14;
              PP = 7;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-22;
              PP = 8;
            }
            else
            {
              continue;
            }
          }

          if ((cr==2) && (bd==0))
          {
            if ((ch>=0) && (ch<8))
            {
              TT = ch+24;
              PP = 1;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+16;
              PP = 2;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch+8;
              PP = 3;
            }
            else if (ch>23)
            {
              TT = ch;
              PP = 4;
            }
          }

          if ((cr==2) && (bd==1))
          {
            if ((ch>=0) && (ch<8))
            {
              TT = ch+24;
              PP = 5;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+16;
              PP = 6;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch+8;
              PP = 7;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch;
              PP = 8;
            }
          }

          if ((cr==2) && (bd==3))
          {
             if ((ch>=0) && (ch<8))
            {
              TT = ch+24;
              PP = 9;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+16;
              PP = 10;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch+8;
              PP = 11;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch;
              PP = 12;
            }
          }

          if ((cr==2) && (bd==4))
          {
            if ((ch>=0) && (ch<8))
            {
              TT = ch+16;
              PP = 1;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+8;
              PP = 2;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch;
              PP = 3;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch-8;
              PP = 4;
            }
          }

          if ((cr==2) && (bd==6))
          {
            if ((ch>=0) && (ch<8))
            {
              TT = ch+16;
              PP = 5;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+8;
              PP = 6;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch;
              PP = 7;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch-8;
              PP = 8;
            }
          }

          if ((cr==2) && (bd==7))
          {
            if ((ch>=0) && (ch<8))
            {
              TT = ch+16;
              PP = 9;
            }
            else if ((ch>7) && (ch<16))
            {
              TT = ch+8;
              PP = 10;
            }
            else if ((ch>15) && (ch<24))
            {
              TT = ch;
              PP = 11;
            }
            else if ((ch>23) && (ch<32))
            {
              TT = ch-8;
              PP = 12;
            }
          }
 
          if ((cr==2) && (bd==9))
          {
            if ((ch>-1) && (ch<4))
            {
              TT = ch+6;
              PP = 1;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-2;
              PP = 2;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-10;
              PP = 3;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-18;
              PP = 4;
            }
            else
            {
              continue;
            }
          }

          if ((cr==2) && (bd==10))
          {
            if ((ch>-1) && (ch<4))
            {
              TT = ch+6;
              PP = 5;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-2;
              PP = 6;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-10;
              PP = 7;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-18;
              PP = 8;
            }
            else
            {
              continue;
            }
          }

          if ((cr==2) && (bd==12))
          {
            if ((ch>-1) && (ch<4))
            {
              TT = ch+6;
              PP = 9;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-2;
              PP = 10;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-10;
              PP = 11;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-18;
              PP = 12;
            }
            else
            {
              continue;
            }
          }

          if ((cr==2) && (bd==13))
          {
            if ((ch>-1) && (ch<4))
            {
              TT = ch+2;
              PP = 1;
            }           
            else if ((ch>7) && (ch<12))
            {
              TT = ch-6;
              PP = 2;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-14;
              PP = 3;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-22;
              PP = 4;
            }
            else
            {
              continue;
            }
          }

          if ((cr==2) && (bd==14))
          {
            if ((ch>=0) && (ch<4))
            {
              TT = ch+2;
              PP = 5;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-6;
              PP = 6;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-14;
              PP = 7;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-22;
              PP = 8;
            }
            else
            {
              continue;
            }
          }
          
          if ((cr==3) && (bd==0))
          {
            if ((ch>=0) && (ch<6))
            {
              TT = ch+10;
              PP = 1;
            }
            else if ((ch>5) && (ch<12))
            {
              TT = ch+4;
              PP = 2;
            }
            else if ((ch>11) && (ch<18))
            {
              TT = ch-2;
              PP = 3;
            }
            else if ((ch>17) && (ch<24))
            {
              TT = ch-8;
              PP = 4;
            }
            else if ((ch>23) &&(ch<30))
            {
              TT = ch-14;
              PP = 5;
            }
            else if ((ch>29) && (ch<32))
            {
              TT = ch-20;
              PP = 6;
            }
            else
            {
              continue;
            }
          }
          
          if ((cr==3) && (bd==1))
          {
            if ((ch>=0) && (ch<4))
            {
              TT = ch+12;
              PP = 6;
            }
            else if ((ch>3) && (ch<10))
            {
              TT = ch+6;
              PP = 7;
            }
            else if ((ch>9) && (ch<16))
            {
              TT = ch;
              PP = 8;
            }
            else if ((ch>15) && (ch<22))
            {
              TT = ch-6;
              PP = 9;
            }
            else if ((ch>21) &&(ch<28))
            {
              TT = ch-12;
              PP = 10;
            }
            else if ((ch>27) && (ch<32))
            {
              TT = ch-18;
              PP = 11;
            }
            else
            {
              continue;
            }
          }
                  
          if ((cr==3) && (bd==3))
          {
            if ((ch>=0) && (ch<2))
            {
              TT = ch+14;
              PP = 11;
            }
            else if ((ch>1) && (ch<8))
            {
              TT = ch+8;
              PP = 12;
            }
            else
            {
              continue;
            }
          }
          
          if ((cr==3) && (bd==4))
          {
            if ((ch>=0) && (ch<4))
            {
              TT = ch+2;
              PP = 9;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-6;
              PP = 10;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-14;
              PP = 11;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-22;
              PP = 12;
            }
            else
            {
              continue;
            }
          }
          
          if ((cr==3) && (bd==6))
          {
            TT = 1;
            if ((ch>=0) && (ch<4))
            {
              PP = ch+1;
            }
            else if ((ch>7) && (ch<12))
            {
              PP = ch-3;
            }
            else if ((ch>15) && (ch<20))
            {
              PP = ch-7;
            }
            else
            {
              continue;
            }
          }

          if ((cr==3) && (bd==7))
          {
            ew = 1;
            if ((ch>=0) && (ch<6))
            {
              TT = ch+10;
              PP = 1;
            }
            else if ((ch>5) && (ch<12))
            {
              TT = ch+4;
              PP = 2;
            }
            else if ((ch>11) && (ch<18))
            {
              TT = ch-2;
              PP = 3;
            }
            else if ((ch>17) && (ch<24))
            {
              TT = ch-8;
              PP = 4;
            }
            else if ((ch>23) &&(ch<30))
            {
              TT = ch-14;
              PP = 5;
            }
            else if ((ch>29) && (ch<32))
            {
              TT = ch-20;
              PP = 6;
            }
            else
            {
              continue;
            }
          }

          if ((cr==3) && (bd==9))
          {
            ew = 1;
            if ((ch>=0) && (ch<4))
            {
              TT = ch+12;
              PP = 6;
            }
            else if ((ch>3) && (ch<10))
            {
              TT = ch+6;
              PP = 7;
            }
            else if ((ch>9) && (ch<16))
            {
              TT = ch;
              PP = 8;
            }
            else if ((ch>15) && (ch<22))
            {
              TT = ch-6;
              PP = 9;
            }
            else if ((ch>21) &&(ch<28))
            {
              TT = ch-12;
              PP = 10;
            }
            else if ((ch>27) && (ch<32))
            {
              TT = ch-18;
              PP = 11;
            }
            else
            {
              continue;
            }
          }

          if ((cr==3) && (bd==10))
          {
            ew = 1;
            if ((ch>=0) && (ch<2))
            {
              TT = ch+14;
              PP = 11;
            }
            else if ((ch>1) && (ch<8))
            {
              TT = ch+8;
              PP = 12;
            }
            else
            {
              continue;
            }
          }

          if ((cr==3) && (bd==12))
          {
            ew = 1;
            if ((ch>-1) && (ch<4))
            {
              TT = ch+2;
              PP = 9;
            }
            else if ((ch>7) && (ch<12))
            {
              TT = ch-6;
              PP = 10;
            }
            else if ((ch>15) && (ch<20))
            {
              TT = ch-14;
              PP = 11;
            }
            else if ((ch>23) && (ch<28))
            {
              TT = ch-22;
              PP = 12;
            }
            else
            {
              continue;
            }
          }

          if ((cr==3) && (bd==13))
          {
            TT = 1;
            ew = 1;
            if ((ch>=0) && (ch<4))
            {
              PP = ch+1;
            }
            else if ((ch>7) && (ch<12))
            {
              PP = ch-3;
            }
            else if ((ch>15) && (ch<20))
            {
              PP = ch-7;
            }
            else
            {
              continue;
            }
          }

        TH1D* ADC = (TH1D*)in->Get(Form("crate_%d_adc_bd%d_ch%d_prepost0",cr,bd,ch));
        MyFile->cd();
        ADC->Write(Form("AdcEW%dPP%dTT%d",ew,PP,TT)); //SetName(Form("AdcEW%dPP%dTT%d",ew,PP,TT));
        in->cd();
        //Form("AdcEW%dPP%dTT%d",ew,PP,TT)
    }
}
}
}