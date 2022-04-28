import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp2d

# First, import the data for all nMIP and refMult values (from ROOT).
data = np.loadtxt(r'D:\27gev_production\data\poly.txt')
index = int(len(data)/17)
data = data.reshape(index, 17)

# These are the "unweighted" weights. (We'll add more complex ones later.)
w = np.ones(16)

# Scale the data by the weights.
data1 = w*data[:, 1:]

# Now let's make an array of the values, with weights.
mipVref = np.empty((2, index))
mipVref[1, ] = data[:, 0]

for j in range(index):
    mipVref[0, j] += np.sum(data1[j])

# Plot the array.
# H, xedges, yedges = np.histogram2d(mipVref[0], mipVref[1], bins=[100, 100])
# ,cmap=plt.cm.get_cmap("hot"))

# Figure out centers of bins


# def centers(edges):
#     return edges[:-1] + np.diff(edges[:2])/2


# xcenters = centers(xedges)
# ycenters = centers(yedges)

# Construct interpolator
# pdf = interp2d(xcenters, ycenters, H)

plt.hist2d(mipVref[0], mipVref[1], bins=[
           150, 150], cmap=plt.cm.get_cmap("jet"))
#plt.pcolor(xedges, yedges, pdf(xedges, yedges), cmap=plt.cm.get_cmap("hot"))
plt.title("RefMult vs nMIP, Weight 0")
plt.xlabel("nMIP")
plt.ylabel("refMult")
plt.colorbar()
plt.show()
