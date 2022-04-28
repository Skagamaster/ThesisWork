TRandom3* tr3 = new TRandom3;

int CompressedId(int ew, int pp, int tt){
  int id = pp*31+tt;
  return (ew>0)?id:-id;
}

int ProblemColor(int run, int ew, int pp, int tt){
  // you will need to implement this yourself, obviously, by reading from database or your own text file or whatever
  double xran = tr3->Uniform(1.0);
  int level;
  if (xran < 0.85) level=1;
  else if (xran < 0.95) level=2;
  else level=3;
  return level;
}

void TileProblemSummary(int FirstRun=1901, int LastRun=1933){

  gStyle->SetOptStat(0);
  int ProblemPalette[4] = {0,kGreen,kYellow,kRed};
  gStyle->SetPalette(4,ProblemPalette);

  int Nruns=LastRun-FirstRun+1;
  int maxId = CompressedId(1,12,31);
  int minId = CompressedId(-1,12,31);
  TH2I* Sum = new TH2I("ProblemSummary",Form("Problem Summary for runs %d-%d",FirstRun,LastRun),maxId-minId+1,minId-0.5,maxId+0.5,Nruns,FirstRun-0.5,LastRun+0.5);
  Sum->GetXaxis()->SetTitle("Compressed Tile ID");
  Sum->GetYaxis()->SetTitle("Run Number");
  for (int run=FirstRun; run<=LastRun; run++){
    for (int ew=0; ew<2; ew++){
      for (int pp=1; pp<=12; pp++){
	for (int tt=1; tt<=31; tt++){
	  Sum->Fill(CompressedId(ew,pp,tt),run,ProblemColor(run,ew,pp,tt));
	}
      }
    }
  }
  TCanvas* tc = new TCanvas("Probs","Probs",1500,600);
  tc->Draw();
  Sum->Draw("colz");
}

  
