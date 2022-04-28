from MC_glauber import *
import time
import numpy as np
from numpy.random import default_rng
rng = default_rng()
time_prime = time.perf_counter()

N = 100
Particle1 = '197Au'
Particle2 = '197Au'
A1 = 197
A2 = 197
Energy = 14.5  # GeV
bRange = 1.1
model1 = '2pF'
model2 = '2pF'
Range = 2
Bins = 100

j1 = ((pNucleus == Particle1) & (pModel == model1))
j2 = ((pNucleus == Particle2) & (pModel == model2))
j1 = np.hstack(np.where(j1 == True))
j2 = np.hstack(np.where(j2 == True))
if len(j1) > 0:
    i1 = j1[0]
if len(j2) > 0:
    i2 = j2[0]
Rp1 = float(pr2[i1])
Rp2 = float(pr2[i2])
r1 = np.linspace(0, Rp1, Bins)
r2 = np.linspace(0, Rp2, Bins)

# Create a weighted, random array of impact parameters (where it is more likely to
# have a peripheral rather than a central collision).
b_range = np.linspace(0, (Rp1 + Rp2) * bRange, int(1e6))  # 1M possible b values to pull from
b_prob = b_range/np.linalg.norm(b_range, 1)  # linear weighting
b = rng.choice(b_range, N, p=b_prob)
Npart = np.zeros(N, float)
Ncoll = np.zeros(N, float)
Maxr = np.sqrt(SigI(Energy) / np.pi)

# Now we want to make sure the nucleus does not have overlapping necleons. This can easily be done with a loop,
# but that's devastatingly slow for any real Monte Carlo numbers (like on the order of 10^6+).
counter = 0
time_start = time.perf_counter()
Nucleus_test = []
N_cp = []
for L in range(N):
    block0 = time.perf_counter()
    Nucleus_test.append([])
    N_cp.append(np.zeros((197, 197)))
    if L % 50 == 49:
        print("Ion collision", L+1, "of", N, "in progress.")
    Nucleus1 = np.zeros((A1, 7), float)
    Nucleus2 = np.zeros((A2, 7), float)
    # Gives each nucleon is own radial distance from the center of the nucleus
    # Sampled from the NCD function and distributed with a factor 4*pi*r^2*p(r)
    Nucleus1[:, 0] = distribute1D(r1, NCD(Particle1, model1, Range, Bins), A1)
    Nucleus2[:, 0] = distribute1D(r2, NCD(Particle2, model2, Range, Bins), A2)
    # Nucleons are then given random azimuthal distances such that no particular point is more likely to be
    # populated than another.
    # Theta coordinates
    Nucleus1[:, 1] = np.arccos(np.multiply(2, rng.random(A1)) - 1)
    Nucleus2[:, 1] = np.arccos(np.multiply(2, rng.random(A2)) - 1)
    # Phi coordinates
    Nucleus1[:, 2] = np.multiply(2 * np.pi, rng.random(A1))
    Nucleus2[:, 2] = np.multiply(2 * np.pi, rng.random(A2))
    # Cartesian coordinates (x,y,z) are determined from spherical coordinates and passed to the nuclei arrays
    # x coordinates
    Nucleus1[:, 3] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])), np.cos(Nucleus1[:, 2]))
    Nucleus2[:, 3] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])), np.cos(Nucleus2[:, 2]))
    # y coordinates
    Nucleus1[:, 4] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])), np.sin(Nucleus1[:, 2]))
    Nucleus2[:, 4] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])), np.sin(Nucleus2[:, 2]))
    # z coordinates
    Nucleus1[:, 5] = np.multiply(Nucleus1[:, 0], np.cos(Nucleus1[:, 1]))
    Nucleus2[:, 5] = np.multiply(Nucleus2[:, 0], np.cos(Nucleus2[:, 1]))

    Nucleus1_5 = np.copy(Nucleus1)
    Nucleus2_5 = np.copy(Nucleus2)

    block1 = time.perf_counter()

    # This is where the real slowdown seems to be.

    deltas1 = np.sqrt(np.power((Nucleus1[:, 3] - Nucleus1[:, 3]), 2) +
                      np.power((Nucleus1[:, 4] + Nucleus1[:, 4]), 2) +
                      np.power((Nucleus1[:, 5] + Nucleus1[:, 5]), 2))
    deltas2 = np.sqrt(np.power((Nucleus2[:, 3] - Nucleus2[:, 3]), 2) +
                      np.power((Nucleus2[:, 4] + Nucleus2[:, 4]), 2) +
                      np.power((Nucleus2[:, 5] + Nucleus2[:, 5]), 2))
    failure = 0
    while np.any(deltas1) < Maxr:
        too_close = len(deltas1[deltas1 < Maxr])
        new_rands1 = np.arccos(np.multiply(2, rng.random(A1)) - 1)
        new_rands2 = np.multiply(2 * np.pi, rng.random(A1))
        Nucleus1[:, 1] = np.where(deltas1 < Maxr, new_rands1, Nucleus1[:, 1])
        Nucleus1[:, 2] = np.where(deltas1 < Maxr, new_rands2, Nucleus1[:,2])
        Nucleus1[:, 3] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])),
                                     np.cos(Nucleus1[:, 2]))
        Nucleus1[:, 4] = np.multiply(np.multiply(Nucleus1[:, 0], np.sin(Nucleus1[:, 1])),
                                     np.sin(Nucleus1[:, 2]))
        Nucleus1[:, 5] = np.multiply(Nucleus1[:, 0], np.cos(Nucleus1[:, 1]))
        deltas1 = np.sqrt(np.power((Nucleus1[:, 3] - Nucleus1[:, 3]), 2) +
                          np.power((Nucleus1[:, 4] + Nucleus1[:, 4]), 2) +
                          np.power((Nucleus1[:, 5] + Nucleus1[:, 5]), 2))
        failure += 1
        if failure > 10:
            Nucleus1[:, 0] = np.where(deltas1 < Maxr, distribute1D(r1, NCD(Particle1, model1, Range, Bins), A1),
                                      Nucleus1[:, 0])
            break
    failure = 0
    while np.any(deltas2) < Maxr:
        too_close = len(deltas2[deltas2 < Maxr])
        new_rands1 = np.arccos(np.multiply(2, rng.random(A2)) - 1)
        new_rands2 = np.multiply(2 * np.pi, rng.random(A2))
        Nucleus2[:, 1] = np.where(deltas1 < Maxr, new_rands1, Nucleus2[:, 1])
        Nucleus2[:, 2] = np.where(deltas1 < Maxr, new_rands2, Nucleus2[:, 2])
        Nucleus2[:, 3] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])),
                                     np.cos(Nucleus2[:, 2]))
        Nucleus2[:, 4] = np.multiply(np.multiply(Nucleus2[:, 0], np.sin(Nucleus2[:, 1])),
                                     np.sin(Nucleus2[:, 2]))
        Nucleus2[:, 5] = np.multiply(Nucleus2[:, 0], np.cos(Nucleus2[:, 1]))
        deltas2 = np.sqrt(np.power((Nucleus2[:, 3] - Nucleus2[:, 3]), 2) +
                          np.power((Nucleus2[:, 4] + Nucleus2[:, 4]), 2) +
                          np.power((Nucleus2[:, 5] + Nucleus2[:, 5]), 2))
        failure += 1
        if failure > 10:
            Nucleus2[:, 0] = np.where(deltas1 < Maxr, distribute1D(r1, NCD(Particle1, model1, Range, Bins), A1),
                                      Nucleus2[:, 0])
            break
    block1_1 = time.perf_counter()
    # print("New code:", block1_1-block1)
    """
    for p1 in range(A1):
        for p1x in range(A1):
            FailSafe = 0  # Prevents program from running indefinitely in some cases
            if p1x == p1:
                pass
            else:
                while np.sqrt(
                        (Nucleus1[p1, 3] - Nucleus1[p1x, 3]) ** 2 + (Nucleus1[p1, 4] + Nucleus1[p1x, 4]) ** 2 + (
                                Nucleus1[p1, 5] + Nucleus1[p1x, 5]) ** 2) < Maxr:
                    Nucleus1[p1x, 1] = np.arccos(2 * rng.random(1) - 1)
                    Nucleus1[p1x, 2] = 2 * np.pi * rng.random(1)
                    Nucleus1[p1x, 3] = Nucleus1[p1x, 0] * np.sin(Nucleus1[p1x, 1]) * np.cos(Nucleus1[p1x, 2])
                    Nucleus1[p1x, 4] = Nucleus1[p1x, 0] * np.sin(Nucleus1[p1x, 1]) * np.sin(Nucleus1[p1x, 2])
                    Nucleus1[p1x, 5] = Nucleus1[p1x, 0] * np.cos(Nucleus1[p1x, 1])
                    FailSafe += 1
                    if FailSafe > 10:
                        Nucleus1[p1x, 0] = distribute1D(r1, NCD(Particle1, model1, Range, Bins), A1)[p1x]
    block1_1 = time.perf_counter()
    print("New code:", block1_1-block1)
    
    for p2 in range(A2):
        for p2x in range(A2):
            FailSafe = 0
            if p2x == p2:
                pass
            else:
                while np.sqrt(
                        (Nucleus2[p2, 3] - Nucleus2[p2x, 3]) ** 2 + (Nucleus2[p2, 4] - Nucleus2[p2x, 4]) ** 2 + (
                                Nucleus2[p2, 5] - Nucleus2[p2x, 5]) ** 2) < Maxr:
                    Nucleus2[p2x, 1] = np.arccos(2 * rng.random(1) - 1)
                    Nucleus2[p2x, 2] = 2 * np.pi * rng.random(1)
                    Nucleus2[p2x, 3] = Nucleus2[p2x, 0] * np.sin(Nucleus2[p2x, 1]) * np.cos(Nucleus2[p2x, 2])
                    Nucleus2[p2x, 4] = Nucleus2[p2x, 0] * np.sin(Nucleus2[p2x, 1]) * np.sin(Nucleus2[p2x, 2])
                    Nucleus2[p2x, 5] = Nucleus2[p2x, 0] * np.cos(Nucleus2[p2x, 1])
                    FailSafe += 1
                    if FailSafe > 10:
                        Nucleus2[p2x, 0] = distribute1D(r2, NCD(Particle2, model2, Range, Bins), A2)[p2x]
    block1_2 = time.perf_counter()
    print("Old code:", block1_2-block1_1)
    """
    Nucleus_test[L].append(Nucleus1[:, 3])
    Nucleus_test[L].append(Nucleus2[:, 3])
    Nucleus_test[L].append(Nucleus1[:, 4])
    Nucleus_test[L].append(Nucleus2[:, 4])

    # This block can be used to make it faster when colliding the same species.
    nuctile1 = np.tile(Nucleus1[:, 3], (A2, 1)).T
    nuctile2 = np.tile(Nucleus2[:, 3], (A1, 1))
    nuctile3 = np.tile(Nucleus1[:, 4], (A2, 1)).T
    nuctile4 = np.tile(Nucleus2[:, 4], (A1, 1))
    Ncoll_builder = np.sqrt(np.add(np.power(np.subtract(nuctile1, nuctile2), 2),
                                   np.power(np.add(-b[L] - nuctile4, nuctile3), 2)))
    Ncoll_builder = np.where(Ncoll_builder.T <= Maxr, 1, 0)
    Ncoll[L] = np.sum(Ncoll_builder)
    Nucleus1[:, 6] = np.where(np.sum(Ncoll_builder, axis=0) == 0, 1, 0)
    Nucleus2[:, 6] = np.where(np.sum(Ncoll_builder, axis=1) == 0, 1, 0)
    # Number of participating particles is the total number of particles minus all the flagged unwounded particles
    Npart[L] = A1 + A2 - np.sum(Nucleus1[:, 6]) - np.sum(Nucleus2[:, 6])
    if model1 == 'FB':
        Rp1 = float(pr2[i1])
    if model2 == 'FB':
        Rp2 = float(pr2[i2])
    # print("Total loop time:", time.perf_counter() - block0)

print("Total run time:", time.perf_counter()-time_prime)

plt.hist(Ncoll, histtype='step', bins=100)
# plt.show()

plt.hist(Npart, histtype='step', bins=100)
# plt.show()

Nucleus_test = np.array(Nucleus_test, dtype="object")
N_cp = np.array(N_cp)
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\N_cp_loop.npy", N_cp)
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\Ncoll_loop.npy", Ncoll)
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\Npart_loop.npy", Npart)
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\Nucleus_test.npy", Nucleus_test)
np.save(r"C:\Users\dansk\Documents\Thesis\Tristan\b_loop.npy", b)
