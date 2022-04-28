# \brief This is an unsupervised learning program designed to
#        find particles in Au+Au collisions at the STAR
#        experiment.
#
# \author Skipper Kagamaster
# \date 03/10/2021
# \email skk317@lehigh.edu
# \affiliation Lehigh University
#
#

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import get_particles as gp

time_start = time.perf_counter()
os.chdir(r"E:\2019Picos\14p5GeV\Runs")

particles = gp.ParticleFinder()
print("Importing data.")
time_1 = time.perf_counter()
particles.data_in(file_loc="20125057.root")
print("Complete:", time.perf_counter()-time_1, "s")
GMM = []
print("Building GMM model 1.")
time_2 = time.perf_counter()
particles.model_gmm(switch=3, n_clusters=5, n_init=10, max_iter=200)
GMM.append(particles.gmm)
print("Complete:", time.perf_counter()-time_2, "s")
print("Building GMM model 2.")
time_3 = time.perf_counter()
particles.model_gmm(switch=3, n_clusters=5, n_init=40, max_iter=200)
GMM.append(particles.gmm)
print("Complete:", time.perf_counter()-time_3, "s")
print("Building GMM model 3.")
time_4 = time.perf_counter()
particles.model_gmm(switch=3, n_clusters=5, n_init=70, max_iter=200)
GMM.append(particles.gmm)
print("Complete:", time.perf_counter()-time_4, "s")

fig, ax = plt.subplots(3, 3, figsize=(16, 9), constrained_layout=True)
cmap_ = plt.cm.get_cmap("jet", 10)
for i in range(3):
    ax[i, 0].scatter(particles.p_g * particles.charge,
                     np.divide(1, particles.beta), s=0.1,
                     c=GMM[i], cmap=cmap_, alpha=0.5)
    ax[i, 1].scatter(particles.p_g * particles.charge,
                     particles.dedx, s=0.1,
                     c=GMM[i], cmap=cmap_, alpha=0.5)
    ax[i, 2].scatter(particles.p_g * particles.charge,
                     particles.m_squared, s=0.1,
                     c=GMM[i], cmap=cmap_, alpha=0.5)
plt.show()
