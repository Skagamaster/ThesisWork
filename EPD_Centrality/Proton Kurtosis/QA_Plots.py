import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fn
import matplotlib.colors as colors

os.chdir(r"C:\Users\dansk\Documents\Thesis\Protons\7p7_GeV\QA")

# Run list
Runs = np.load("run_list.npy", allow_pickle=True)

# Take the averages, with error, and calculate 3 sigma deviations.
df = pd.read_pickle("averages.pkl")

stdRefMult3 = 3*np.std(df["RefMult3"])
aveRefMult3 = np.mean(df["RefMult3"])

stdVz = 3*np.std(df["vZ"])
aveVz = np.mean(df["vZ"])

stdPt = 3*np.std(df["pT"])
avePt = np.mean(df["pT"])

stdPhi = 3*np.std(df["phi"])
avePhi = np.mean(df["phi"])

stdVr = 3*np.std(df["vR"])
aveVr = np.mean(df["vR"])

stdZdcX = 3*np.std(df["ZDCx"])
aveZdcX = np.mean(df["ZDCx"])

stdEta = 3*np.std(df["eta"])
aveEta = np.mean(df["eta"])

stdDca = 3*np.std(df["DCA"])
aveDca = np.mean(df["DCA"])

# Let's make some plots!
fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)

# Plot for the average RefMult3
imRefMult3 = ax[0, 0].errorbar(Runs, df["RefMult3"], yerr=df["RefMult3_err"], fmt='ok', ms=0,
                               mfc='None', capsize=1.5, elinewidth=1)
ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::100])
ax[0, 0].tick_params(labelrotation=45, labelsize=7)
ax[0, 0].axhline(aveRefMult3, 0, 1, c='red', label="Mean")
ax[0, 0].axhline(aveRefMult3+stdRefMult3, 0, 1, c='blue', label=r'3$\sigma$')
ax[0, 0].axhline(aveRefMult3-stdRefMult3, 0, 1, c='blue')
ax[0, 0].set_title(r"$<RefMult3>$", fontsize=20)
ax[0, 0].set_xlabel("Run ID", fontsize=10)
ax[0, 0].set_ylabel(r"$<RefMult3>$", fontsize=10)
ax[0, 0].legend()

# Plot for the average v_z
imVz = ax[0, 1].errorbar(Runs, df["vZ"], yerr=df["vZ_err"], fmt='ok', ms=0,
                        mfc='None', capsize=1.5, elinewidth=1)
ax[0, 1].set_xticks(ax[0, 1].get_xticks()[::100])
ax[0, 1].tick_params(labelrotation=45, labelsize=7)
ax[0, 1].axhline(aveVz, 0, 1, c='red', label="Mean")
ax[0, 1].axhline(aveVz+stdVz, 0, 1, c='blue', label=r'3$\sigma$')
ax[0, 1].axhline(aveVz-stdVz, 0, 1, c='blue')
ax[0, 1].set_title(r"$<V_z>$", fontsize=20)
ax[0, 1].set_xlabel("Run ID", fontsize=10)
ax[0, 1].set_ylabel(r"$<V_z>$ (cm)", fontsize=10)
ax[0, 1].legend()

# Plot for the average p_t
imPt = ax[0, 2].errorbar(Runs, df["pT"], yerr=df["pT_err"], fmt='ok', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1)
ax[0, 2].set_xticks(ax[0, 2].get_xticks()[::100])
ax[0, 2].tick_params(labelrotation=45, labelsize=7)
ax[0, 2].axhline(avePt, 0, 1, c='red', label="Mean")
ax[0, 2].axhline(avePt+stdPt, 0, 1, c='blue', label=r'3$\sigma$')
ax[0, 2].axhline(avePt-stdPt, 0, 1, c='blue')
ax[0, 2].set_title(r"$<p_T>$", fontsize=20)
ax[0, 2].set_xlabel("Run ID", fontsize=10)
ax[0, 2].set_ylabel(r"$<p_T> (\frac{GeV}{c})$", fontsize=10)
ax[0, 2].legend()

# Plot for the average phi
imPhi = ax[0, 3].errorbar(Runs, df["phi"], yerr=df["phi_err"], fmt='ok', ms=0,
                          mfc='None', capsize=1.5, elinewidth=1)
ax[0, 3].set_xticks(ax[0, 3].get_xticks()[::100])
ax[0, 3].tick_params(labelrotation=45, labelsize=7)
ax[0, 3].axhline(avePhi, 0, 1, c='red', label="Mean")
ax[0, 3].axhline(avePhi+stdPhi, 0, 1, c='blue', label=r'3$\sigma$')
ax[0, 3].axhline(avePhi-stdPhi, 0, 1, c='blue')
ax[0, 3].set_title(r"$<\phi>$", fontsize=20)
ax[0, 3].set_xlabel("Run ID", fontsize=10)
ax[0, 3].set_ylabel(r"$<\phi>$ (rad)", fontsize=10)
ax[0, 3].legend()

# Plot for the average v_r
imVr = ax[1, 0].errorbar(Runs, df["vR"], yerr=df["vR_err"], fmt='ok', ms=0,
                         mfc='None', capsize=1.5, elinewidth=1)
ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::100])
ax[1, 0].tick_params(labelrotation=45, labelsize=7)
ax[1, 0].axhline(aveVr, 0, 1, c='red', label="Mean")
ax[1, 0].axhline(aveVr+stdVr, 0, 1,
                 c='blue', label=r'3$\sigma$')
ax[1, 0].axhline(aveVr-stdVr, 0, 1, c='blue')
ax[1, 0].set_title(r"$<v_r>$", fontsize=20)
ax[1, 0].set_xlabel("Run ID", fontsize=10)
ax[1, 0].set_ylabel(r"$<v_r>$ (cm)", fontsize=10)
ax[1, 0].legend()

# Plot for the average ZDCx
imZdcX = ax[1, 1].errorbar(Runs, df["ZDCx"], yerr=df["ZDCx_err"], fmt='ok', ms=0,
                         mfc='None', capsize=1.5, elinewidth=1)
ax[1, 1].set_xticks(ax[1, 1].get_xticks()[::100])
ax[1, 1].tick_params(labelrotation=45, labelsize=7)
ax[1, 1].axhline(aveZdcX, 0, 1, c='red', label="Mean")
ax[1, 1].axhline(aveZdcX+stdZdcX, 0, 1, c='blue', label=r'3$\sigma$')
ax[1, 1].axhline(aveZdcX-stdZdcX, 0, 1, c='blue')
ax[1, 1].set_title(r"$<ZDC_x>$", fontsize=20)
ax[1, 1].set_xlabel("Run ID", fontsize=10)
ax[1, 1].set_ylabel(r"$ZDC_x$ (cm)", fontsize=10)
# ax[1, 1].set_ylim(-200, 200)
ax[1, 1].legend()

# Plot for the average eta
imEta = ax[1, 2].errorbar(Runs, df["eta"], yerr=df["eta_err"], fmt='ok', ms=0,
                         mfc='None', capsize=1.5, elinewidth=1)
ax[1, 2].set_xticks(ax[1, 2].get_xticks()[::100])
ax[1, 2].tick_params(labelrotation=45, labelsize=7)
ax[1, 2].axhline(aveEta, 0, 1, c='red', label="Mean")
ax[1, 2].axhline(aveEta+stdEta, 0, 1, c='blue', label=r'3$\sigma$')
ax[1, 2].axhline(aveEta-stdEta, 0, 1, c='blue')
ax[1, 2].set_title(r"$<\eta>$", fontsize=20)
ax[1, 2].set_xlabel("Run ID", fontsize=10)
ax[1, 2].set_ylabel(r"$\eta$", fontsize=10)
# ax[1, 2].set_ylim(-200, 200)
ax[1, 2].legend()

# Plot for the average DCA
imEta = ax[1, 3].errorbar(Runs, df["DCA"], yerr=df["DCA_err"], fmt='ok', ms=0,
                         mfc='None', capsize=1.5, elinewidth=1)
ax[1, 3].set_xticks(ax[1, 3].get_xticks()[::100])
ax[1, 3].tick_params(labelrotation=45, labelsize=7)
ax[1, 3].axhline(aveDca, 0, 1, c='red', label="Mean")
ax[1, 3].axhline(aveDca+stdDca, 0, 1, c='blue', label=r'3$\sigma$')
ax[1, 3].axhline(aveDca-stdDca, 0, 1, c='blue')
ax[1, 3].set_title(r"$<DCA>$", fontsize=20)
ax[1, 3].set_xlabel("Run ID", fontsize=10)
ax[1, 3].set_ylabel(r"$DCA$ (cm)", fontsize=10)
# ax[1, 3].set_ylim(-200, 200)
ax[1, 3].legend()

plt.show()
plt.close()

# Now for the 1D histogram plots.

# First, let's import the data.
p_t = np.load("pT.npy", allow_pickle=True)
phi = np.load("phi.npy", allow_pickle=True)
dca = np.load("dca.npy", allow_pickle=True)
eta = np.load("eta.npy", allow_pickle=True)
nHitsFit_q = np.load("nhitsfit_charge.npy", allow_pickle=True)
nHits_dEdX = np.load("nhits_dedx.npy", allow_pickle=True)
v_z = np.load("vZ.npy", allow_pickle=True)

# We'll graph v_z separately, for formatting concerns.
fig, ax = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)

ax[0, 0].plot(p_t[1], p_t[0])
ax[0, 0].set_xlabel(r"$p_T$ $(\frac{GeV}{c})$")
ax[0, 0].set_ylabel("Counts")
ax[0, 0].set_yscale('log')

ax[0, 1].plot(phi[1], phi[0])
ax[0, 1].set_xlabel(r"$\phi$")
ax[0, 1].set_ylabel("Counts")
ax[0, 1].set_yscale('log')

ax[0, 2].plot(dca[1], dca[0])
ax[0, 2].set_xlabel("DCA (cm)")
ax[0, 2].set_ylabel("Counts")
ax[0, 2].set_yscale('log')

ax[1, 0].plot(eta[1], eta[0])
ax[1, 0].set_xlabel(r"$\eta$")
ax[1, 0].set_ylabel("Counts")
ax[1, 0].set_yscale('log')

ax[1, 1].plot(nHitsFit_q[1], nHitsFit_q[0])
ax[1, 1].set_xlabel("nHitsFit*charge")
ax[1, 1].set_ylabel("Counts")
ax[1, 1].set_yscale('log')

ax[1, 2].plot(nHits_dEdX[1], nHits_dEdX[0])
ax[1, 2].set_xlabel("nHits_dEdX")
ax[1, 2].set_ylabel("Counts")
ax[1, 2].set_yscale('log')

plt.show()
plt.close()

# And finally our 2D histograms.

# Import the 2D histogram data.
v_r = np.load("vR.npy", allow_pickle=True)
v_r_bins = np.load("vR_bins.npy", allow_pickle=True)
RefMult_TOFMult = np.load("refmult_tofmult.npy", allow_pickle=True)
RefMult_TOFMult_bins = np.load("refmult_tofmult_bins.npy", allow_pickle=True)
RefMult_TOFMatch = np.load("refmult_tofmatch.npy", allow_pickle=True)
RefMult_TOFMatch_bins = np.load("refmult_tofmatch_bins.npy", allow_pickle=True)
RefMult_BetaEta = np.load("refmult_beta_eta.npy", allow_pickle=True)
RefMult_BetaEta_bins = np.load("refmult_beta_eta_bins.npy", allow_pickle=True)
m_pq = np.load("m_pq.npy", allow_pickle=True)
m_pq_bins = np.load("m_pq_bins.npy", allow_pickle=True)
beta_p = np.load("beta_p.npy", allow_pickle=True)
beta_p_bins = np.load("beta_p_bins.npy", allow_pickle=True)
dEdX_pq = np.load("dEdX_pq.npy", allow_pickle=True)
dEdX_pq_bins = np.load("dEdX_pq_bins.npy", allow_pickle=True)


# Make the 2D plots (and v_z)

fig, ax = plt.subplots(2, 4, figsize=(16, 9), constrained_layout=True)

# Plot for RefMult3 vs TOFMult
x = np.linspace(0, 1000, 1001)
y1 = 1.22*x - 24.29
y2 = 1.95*x + 75
ax[0, 0].plot(x, y1, c='red', lw=2)
ax[0, 0].plot(x, y2, c='red', lw=2)
X, Y = np.meshgrid(RefMult_TOFMult_bins[1], RefMult_TOFMult_bins[0])
imRefMult_TOFMult = ax[0, 0].pcolormesh(X, Y, RefMult_TOFMult, cmap="jet", norm=colors.LogNorm())
ax[0, 0].set_xlabel("RefMult3", fontsize=10)
ax[0, 0].set_ylabel("TOFMult", fontsize=10)
ax[0, 0].set_title("RefMult3 vs TOFMult", fontsize=20)
ax[0, 0].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_TOFMult, ax=ax[0, 0])

# Plot for RefMult3 vs TOFMatch
y1 = 0.379*x - 8.6
y2 = 0.631*x + 11.69
ax[0, 1].plot(x, y1, c='red', lw=2)
ax[0, 1].plot(x, y2, c='red', lw=2)
X, Y = np.meshgrid(RefMult_TOFMatch_bins[1], RefMult_TOFMatch_bins[0])
imRefMult_TOFMatch = ax[0, 1].pcolormesh(X, Y, RefMult_TOFMatch, cmap="jet", norm=colors.LogNorm())
ax[0, 1].set_xlabel("RefMult3", fontsize=10)
ax[0, 1].set_ylabel("TOFMatch", fontsize=10)
ax[0, 1].set_title("RefMult3 vs TOFMatch", fontsize=20)
ax[0, 1].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_TOFMatch, ax=ax[0, 1])

# Plot for RefMult3 vs beta_eta1
y1 = 0.202*x - 11.5
y2 = 0.37*x + 8
ax[0, 2].plot(x, y1, c='red', lw=2)
ax[0, 2].plot(x, y2, c='red', lw=2)
X, Y = np.meshgrid(RefMult_BetaEta_bins[1], RefMult_BetaEta_bins[0])
imRefMult_BetaEta = ax[0, 2].pcolormesh(X, Y, RefMult_BetaEta, cmap="jet", norm=colors.LogNorm())
ax[0, 2].set_xlabel("RefMult3", fontsize=10)
ax[0, 2].set_ylabel(r"$\beta \eta$1", fontsize=10)
ax[0, 2].set_title(r"RefMult3 vs $\beta \eta$1", fontsize=20)
ax[0, 2].set_ylim(0, np.max(Y))
fig.colorbar(imRefMult_BetaEta, ax=ax[0, 2])

# Plot for v_x vs v_y
circle1 = plt.Circle((-0.27275,-0.235275),2, color = 'red', fill = False, clip_on = False, lw=2)
X, Y = np.meshgrid(v_r_bins[1], v_r_bins[0])
imVr = ax[0, 3].pcolormesh(X, Y, v_r, cmap="jet", norm=colors.LogNorm())
ax[0, 3].set_xlabel(r"$v_x$ (cm)", fontsize=10)
ax[0, 3].set_ylabel(r"$v_y$ (cm)", fontsize=10)
ax[0, 3].set_title(r"$v_r$ Position", fontsize=20)
# ax[0, 3].add_artist(circle1)
# fig.colorbar(imVr, ax=ax[0, 3])

# Plot for mass vs momentum
X, Y = np.meshgrid(m_pq_bins[1], m_pq_bins[0])
imMp = ax[1, 0].pcolormesh(X, Y, m_pq, cmap="jet", norm=colors.LogNorm())
ax[1, 0].set_ylabel(r"$m^2 (\frac{GeV^2}{c^4})$", fontsize=10)
ax[1, 0].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
ax[1, 0].set_title("Mass vs Momentum", fontsize=20)
ax[1, 0].axhline(0.6, c='red', lw=2)
ax[1, 0].axhline(1.2, c='red', lw=2)
fig.colorbar(imMp, ax=ax[1, 0])

# Plot for Beta vs momentum
X, Y = np.meshgrid(beta_p_bins[1], beta_p_bins[0])
imBp = ax[1, 1].pcolormesh(X, Y, beta_p, cmap="jet", norm=colors.LogNorm())
ax[1, 1].set_xlabel(r"p $(\frac{GeV}{c})$", fontsize=10)
ax[1, 1].set_ylabel(r"1/$\beta$", fontsize=10)
ax[1, 1].set_title(r"$\beta$ vs Momentum", fontsize=20)
fig.colorbar(imBp, ax=ax[1, 1])

# Plot for dEdX vs momentum
X, Y = np.meshgrid(dEdX_pq_bins[1], dEdX_pq_bins[0])
imDp = ax[1, 2].pcolormesh(X, Y, dEdX_pq, cmap="jet", norm=colors.LogNorm())
ax[1, 2].set_xlabel(r"p*q $(\frac{GeV}{c})$", fontsize=10)
ax[1, 2].set_ylabel(r"$\frac{dE}{dX} (\frac{KeV}{cm})$", fontsize=10)
ax[1, 2].set_title(r"p*q vs $\frac{dE}{dX}$", fontsize=20)
fig.colorbar(imDp, ax=ax[1, 2])

# Plot for v_z
ax[1, 3].axvline(-30, color='red', lw=2)
ax[1, 3].axvline(30, color='red', lw=2)
ax[1, 3].plot(v_z[1], v_z[0])
ax[1, 3].set_xlabel(r"$v_z$ (cm)")
ax[1, 3].set_ylabel("Counts")
ax[1, 3].set_yscale('log')
ax[1, 3].set_xlim(-70, 70)

plt.show()
plt.close()

# Now let's make the cuts to find the bad runs.
vR_bad_pos = df[df.vR > (aveVr+stdVr)]["vR"].index
vR_bad_neg = df[df.vR < (aveVr-stdVr)]["vR"].index
vZ_bad_pos = df[df.vZ > (aveVz+stdVz)]["vZ"].index
vZ_bad_neg = df[df.vZ < (aveVz-stdVz)]["vZ"].index
pT_bad_pos = df[df.pT > (avePt+stdPt)]["pT"].index
pT_bad_neg = df[df.pT < (avePt-stdPt)]["pT"].index
dca_bad_pos = df[df.DCA > (aveDca+stdDca)]["DCA"].index
dca_bad_neg = df[df.DCA < (aveDca-stdDca)]["DCA"].index
RefMult3_bad_pos = df[df.RefMult3 > (aveRefMult3+stdRefMult3)]["RefMult3"].index
RefMult3_bad_neg = df[df.RefMult3 < (aveRefMult3-stdRefMult3)]["RefMult3"].index
phi_bad_pos = df[df.phi > (avePhi+stdPhi)]["phi"].index
phi_bad_neg = df[df.phi < (avePhi-stdPhi)]["phi"].index
eta_bad_pos = df[df.eta > (aveEta+stdEta)]["eta"].index
eta_bad_neg = df[df.eta < (aveEta-stdEta)]["eta"].index
zdc_bad_pos = df[df.ZDCx > (aveZdcX+stdZdcX)]["ZDCx"].index
zdc_bad_neg = df[df.ZDCx < (aveZdcX-stdZdcX)]["ZDCx"].index
u, c = np.unique(np.hstack((vR_bad_pos, vR_bad_neg, vZ_bad_pos, vZ_bad_neg, pT_bad_pos,
                            pT_bad_neg, dca_bad_pos, dca_bad_neg, RefMult3_bad_pos,
                            RefMult3_bad_neg, phi_bad_pos, phi_bad_neg, eta_bad_pos,
                            eta_bad_neg, zdc_bad_pos, zdc_bad_neg)),
                 return_counts=True)
print(Runs[u[c > 1]])
badRuns = np.asarray(Runs[u[c > 1]])
# np.save("badRuns.npy", badRuns)
