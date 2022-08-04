from MC_glauber import *
import time
import numpy as np

# Imports the main library file

# DisplayData()

N = 10
Particle1 = '197Au'
Particle2 = '197Au'
A1 = 197
A2 = 197
Energy = 14.6  # GeV
bRange = 0.5
model1 = '2pF'
model2 = '2pF'
Range = 2
Bins = 100

time_start = time.perf_counter()
# help(Collider)  # Shows the documentation for the Collider function
b, Nucleus1, Nucleus2, Npart, Ncoll, Maxr, Rp1, Rp2 = Collider(N, Particle1, A1, Particle2, A2, model1, model2,
                                                               Energy, bRange, Range, Bins)

"""
This next section saves the generated b, N_coll, and N_part arrays as both *.npy files for Python anaylsis and
as *.txt files for any other analysis (like in ROOT). Files are formatted with a suffix of:
Species1 _ Species2 _ Collision energy _ Number of collisions
"""
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\Ncoll_{}_{}_{}_{}.npy".format(Particle1, Particle2, Energy, N),
        Ncoll)
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\Npart_{}_{}_{}_{}.npy".format(Particle1, Particle2, Energy, N),
        Npart)
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\b_{}_{}_{}_{}.npy".format(Particle1, Particle2, Energy, N), b)
np.savetxt(r"C:\Users\dansk\Documents\Thesis\Tristan\Ncoll_{}_{}_{}_{}.txt".format(Particle1, Particle2, Energy, N),
           Ncoll)
np.savetxt(r"C:\Users\dansk\Documents\Thesis\Tristan\Npart_{}_{}_{}_{}.txt".format(Particle1, Particle2, Energy, N),
           Npart)
np.savetxt(r"C:\Users\dansk\Documents\Thesis\Tristan\b_{}_{}_{}_{}.txt".format(Particle1, Particle2, Energy, N), b)

time_stamp_1 = time.perf_counter()
print("Collider run time:", time.perf_counter()-time_start)

# help(PlotNuclei)
PlotNuclei(Nucleus1, Nucleus2, Particle1, Particle2, model1, model2, Rp1, Rp2, Range, Bins)

# help(ShowCollision)
ShowCollision(N, Particle1, A1, Particle2, A2, Rp1, Rp2, Nucleus1, Nucleus2, b, Npart, Ncoll, Maxr)

# help(PlotResults)
PlotResults(b, Npart, Ncoll, Particle1, Particle2, N, Energy, 10)

'''
N = 100
Particle1 = '63Cu'
Particle2 = '63Cu'
A1 = 63
A2 = 63
Energy = 200
bRange = 1.1
model1 = '2pF'
model2 = '2pF'
Range = 2
Bins = 100
b, Nucleus1, Nucleus2, Npart, Ncoll, Maxr, Rp1, Rp2 = Collider(N, Particle1, A1, Particle2, A2, model1, model2, Energy,
                                                               bRange, Range, Bins)
PlotResults(b, Npart, Ncoll, Particle1, Particle2, N, Energy, 22)
'''
