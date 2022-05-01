import numpy as np
import pandas as pd
from scipy.stats import moment
from scipy.stats import skew, kurtosis
import warnings

# warnings.filterwarnings("ignore", category=RuntimeWarning)


def proton_cumu_array(proton_df, refmult_df, ref_quant, name='refmult', pid=1):
    print("Making cumulant array for", name)
    arr = []
    index = refmult_df[(refmult_df[name] >= 0) & (refmult_df[name] < ref_quant[0])].index.to_numpy()
    df = proton_df.loc[index]
    df = df[(abs(df['pid']) == pid) & (df['p_t'] <= 0.8) & (df['p_t'] >= 0.4)
            & (abs(df['q']) == 1.0) & (abs(df['eta']) <= 1.0)
            & (abs(df['rap']) <= 0.5)]
    df['net'] = df['pid']*df['q']
    arr.append(df['net'].sum(level=0).to_numpy())
    print("Step 1")
    for i in range(len(ref_quant)-1):
        print("Step", i+2)
        index = refmult_df[(refmult_df[name] >= ref_quant[i]) & (refmult_df[name] < ref_quant[i+1])].index.to_numpy()
        df = proton_df.loc[index]
        df = df[(abs(df['pid']) == pid) & (df['p_t'] <= 0.8) & (df['p_t'] >= 0.4)
                & (abs(df['q']) == 1.0) & (abs(df['eta']) <= 1.0)
                & (abs(df['rap']) <= 0.5)]
        df['net'] = df['pid'] * df['q']
        arr.append(df['net'].sum(level=0).to_numpy())
    print("Last step")
    index = refmult_df[(refmult_df[name] >= ref_quant[-1])].index.to_numpy()
    df = proton_df.loc[index]
    df = df[(abs(df['pid']) == pid) & (df['p_t'] <= 0.8) & (df['p_t'] >= 0.4)
            & (abs(df['q']) == 1.0) & (abs(df['eta']) <= 1.0)
            & (abs(df['rap']) <= 0.5)]
    df['net'] = df['pid'] * df['q']
    arr.append(df['net'].sum(level=0).to_numpy())
    cumulants = np.empty((4, len(ref_quant)+1))
    for i in range(len(cumulants[0])):
        cumulants[0][i] = np.mean(arr[i])
        cumulants[1][i] = moment(arr[i], moment=2)
        cumulants[2][i] = moment(arr[i], moment=3)/cumulants[1][i]
        cumulants[3][i] = (moment(arr[i], moment=4) - 3*np.power(moment(arr[i], moment=2), 2)) / \
                          cumulants[1][i]
        cumulants[1][i] = cumulants[1][i]/cumulants[0][i]
    return cumulants


def proton_cumu_array_cbwc(proton_df, refmult_df, ref_quant, name='refmult', pid=1):
    print("Making CBWC cumulant array for", name + '.')
    arr = [[], [], [], []]
    lens = []
    values = np.unique(refmult_df[name].to_numpy())
    err_arr = []
    count = 0
    for i in values:
        if count % 100 == 0:
            print(count, "of", len(values))
        count += 1
        index = refmult_df[refmult_df[name] == i].index.to_numpy()
        df = proton_df.loc[index]
        df = df[(abs(df['pid']) == pid) & (df['p_t'] <= 0.8) & (df['p_t'] >= 0.4)
                & (abs(df['q']) == 1.0) & (abs(df['eta']) <= 1.0)
                & (abs(df['rap']) <= 0.5)]
        df['net'] = df['pid'] * df['q']
        protons = df['net'].sum(level=0).to_numpy()
        lens.append(len(protons))
        err_arr.append(moment(protons, moment=2))
        arr[0].append(np.mean(protons))
        arr[1].append(moment(protons, moment=2)/np.mean(protons))
        arr[2].append(moment(protons, moment=3) / moment(protons, moment=2))
        arr[3].append((moment(protons, moment=4) - 3 * np.power(moment(protons, moment=2), 2)) /
                      moment(protons, moment=2))
    arr = np.asarray(arr)
    lens = np.asarray(lens)
    err_arr = np.asarray(err_arr)
    arr[0] = np.multiply(arr[0], lens)
    arr[1] = np.multiply(arr[1], lens)
    arr[2] = np.multiply(arr[2], lens)
    arr[3] = np.multiply(arr[3], lens)

    # Apply the CBWC
    cumulants = np.zeros((4, len(ref_quant)+1))
    cumulant_err = np.zeros((4, len(ref_quant)+1))

    index = values <= ref_quant[0]
    for i in range(4):
        c_pro = np.sum(arr[i][index])/np.sum(lens[index])
        cumulants[i][0] = c_pro
    cumulant_err[0][0] = np.sum(np.sqrt(err_arr[index]/lens[index]))/np.sum(lens[index])
    cumulant_err[2][0] = np.sum(np.sqrt(6*err_arr[index]/lens[index]))/np.sum(lens[index])
    cumulant_err[3][0] = np.sum(np.sqrt(24 * np.power(err_arr[index], 2)/lens[index]))/np.sum(lens[index])
    for i in range(len(ref_quant)-1):
        index = (values > ref_quant[i]) & (values <= ref_quant[i+1])
        for j in range(4):
            c_pro = np.sum(arr[j][index])/np.sum(lens[index])
            cumulants[j][i+1] = c_pro
        cumulant_err[0][i] = np.sum(np.sqrt(err_arr[index] / lens[index])) / np.sum(lens[index])
        cumulant_err[2][i] = np.sum(np.sqrt(6 * err_arr[index] / lens[index]))
        cumulant_err[3][i] = np.sum(np.sqrt(24 * np.power(err_arr[index], 2) / lens[index]))
    index = values > ref_quant[-1]
    for i in range(4):
        c_pro = np.sum(arr[i][index]) / np.sum(lens[index])
        cumulants[i][-1] = c_pro
    cumulant_err[0][-1] = np.sum(np.sqrt(err_arr[index]/lens[index]))/np.sum(lens[index])
    cumulant_err[2][-1] = np.sum(np.sqrt(6*err_arr[index]/lens[index]))
    cumulant_err[3][-1] = np.sum(np.sqrt(24 * np.power(err_arr[index], 2) / lens[index]))

    return cumulants, cumulant_err, values


def proton_cumu_array_no_cbwc(proton_df, refmult_df, ref_quant, name='refmult', pid=1):
    print("Making non-CBWC cumulant array for", name + '.')
    arr = [[], [], [], []]
    lens = []
    values = np.unique(refmult_df[name].to_numpy())
    err_arr = []
    count = 0
    for i in values:
        if count % 100 == 0:
            print(count, "of", len(values))
        count += 1
        index = refmult_df[refmult_df[name] == i].index.to_numpy()
        df = proton_df.loc[index]
        df = df[(abs(df['pid']) == pid) & (df['p_t'] <= 0.8) & (df['p_t'] >= 0.4)
                & (abs(df['q']) == 1.0) & (abs(df['eta']) <= 1.0)
                & (abs(df['rap']) <= 0.5)]
        df['net'] = df['pid'] * df['q']
        protons = df['net'].sum(level=0).to_numpy()
        lens.append(len(protons))
        err_arr.append(moment(protons, moment=2))
        arr[0].append(np.mean(protons))
        arr[1].append(moment(protons, moment=2)/np.mean(protons))
        arr[2].append(moment(protons, moment=3) / moment(protons, moment=2))
        arr[3].append((moment(protons, moment=4) - 3 * np.power(moment(protons, moment=2), 2)) /
                      moment(protons, moment=2))
    arr = np.asarray(arr)
    lens = np.asarray(lens)
    err_arr = np.asarray(err_arr)
    arr[0] = np.multiply(arr[0], lens)
    arr[1] = np.multiply(arr[1], lens)
    arr[2] = np.multiply(arr[2], lens)
    arr[3] = np.multiply(arr[3], lens)

    # Apply the CBWC
    cumulants = np.zeros((4, len(ref_quant)+1))
    cumulant_err = np.zeros((4, len(ref_quant)+1))

    index = values <= ref_quant[0]
    for i in range(4):
        c_pro = np.sum(arr[i][index])/np.sum(lens[index])
        cumulants[i][0] = c_pro
    cumulant_err[0][0] = np.sum(np.sqrt(err_arr[index]/lens[index]))/np.sum(lens[index])
    cumulant_err[2][0] = np.sum(np.sqrt(6*err_arr[index]/lens[index]))/np.sum(lens[index])
    cumulant_err[3][0] = np.sum(np.sqrt(24 * np.power(err_arr[index], 2)/lens[index]))/np.sum(lens[index])
    for i in range(len(ref_quant)-1):
        index = (values > ref_quant[i]) & (values <= ref_quant[i+1])
        for j in range(4):
            c_pro = np.sum(arr[j][index])/np.sum(lens[index])
            cumulants[j][i+1] = c_pro
        cumulant_err[0][i] = np.sum(np.sqrt(err_arr[index] / lens[index])) / np.sum(lens[index])
        cumulant_err[2][i] = np.sum(np.sqrt(6 * err_arr[index] / lens[index]))
        cumulant_err[3][i] = np.sum(np.sqrt(24 * np.power(err_arr[index], 2) / lens[index]))
    index = values > ref_quant[-1]
    for i in range(4):
        c_pro = np.sum(arr[i][index]) / np.sum(lens[index])
        cumulants[i][-1] = c_pro
    cumulant_err[0][-1] = np.sum(np.sqrt(err_arr[index]/lens[index]))/np.sum(lens[index])
    cumulant_err[2][-1] = np.sum(np.sqrt(6*err_arr[index]/lens[index]))
    cumulant_err[3][-1] = np.sum(np.sqrt(24 * np.power(err_arr[index], 2) / lens[index]))

    return cumulants, cumulant_err, values
