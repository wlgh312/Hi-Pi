import numpy as np
import glob
def load_dataset2(folderfmt, test_ratio=0.3, verbose=False):
    fnlist = np.array(sorted(glob.glob(folderfmt)))
    print(fnlist)  # 'out/c0' 'out/c1' 'out/c2' 'out/c4' 'out/c5'

    nclasses = len(fnlist)
    if verbose:
        print('nclasses= {}'.format(nclasses))
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []

    ftypes = ('jpg', 'JPG', 'png', 'PNG')

    vidxs = []
    vclas = []
    for i in range(len(fnlist)):
        fndir = fnlist[i]

        listc1 = []
        for ftype in ftypes:
            fnfmt = '{}/*.{}'.format(fndir, ftype)
            if verbose:
                print(fnfmt)
            listc1.extend(sorted(glob.glob(fnfmt)))

        vclas.append(listc1)

        listidx = i * np.ones_like(listc1, dtype=int)
        vidxs.append(listidx)
        # np.array(shuffle(listc1))
        # print(listc1)
        if verbose:
            print(fndir, '------------->', np.shape(listc1))

    return vclas, vidxs