import numpy as np
import uproot as up
import awkward as ak
import matplotlib.pyplot as plt
import os
import time

data = up.open(r"C:\PhysicsProcessing\st_physics_20179015_raw_1000002.picoDst.root")
data = data["PicoDst"]
epdID = data["EpdHit"]["EpdHit.mId"].array()[1]
epd_mQTdata = data["EpdHit"]["EpdHit.mQTdata"].array()
adc_true = np.loadtxt(r"C:\PhysicsProcessing\tiles.txt")


def bin_vec(a):
    def bin_conv(x):
        return int(bin(x)[-11:], 2)
    fun = np.vectorize(bin_conv)
    return fun(a)


def epd_adc_awk(a):
    counts = ak.num(a)
    a_flat = ak.to_numpy(ak.flatten(a))
    a_dtype = int(str(a_flat.dtype)[-2:])
    raw_bytes = a_flat.view(np.uint8)
    raw_bits = np.unpackbits(raw_bytes, bitorder="big")
    reshaped_bits = raw_bits.reshape(-1, a_dtype)
    print(reshaped_bits[0])
    truncated_bits = reshaped_bits[:, :11]
    padded_bits = np.pad(truncated_bits, ((0, 0), (0, 4)))
    as_bytes_again = np.packbits(padded_bits, bitorder="big")
    b = ak.unflatten(as_bytes_again.view(">i2"), counts)
    return b


def epd_adc_test(a):
    counts = ak.num(a)
    a_flat = ak.to_numpy(ak.flatten(a))
    a_flat_ = a_flat.view(np.uint8)[::4]
    b = ak.unflatten(a_flat_, counts)
    return b


time_start = time.perf_counter()
'''
adc_test = []
for i in range(int(len(epd_mQTdata))):
    adc_test.append(bin_vec(epd_mQTdata[i]))
time_start_2 = time.perf_counter()
print(time.perf_counter()-time_start)
'''
adc_test2 = epd_adc_test(epd_mQTdata)
print(time.perf_counter()-time_start)
print(adc_test2[1][:15]==adc_true[:15])
