import numpy as np

patch_size = 5


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset : X.shape[0] + x_offset, y_offset : X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2])
    )
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[
                r - margin : r + margin + 1, c - margin : c + margin + 1
            ]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def data_pre(X, y, train_idx, require_test=True):

    X, y = createImageCubes(X, y, windowSize=patch_size)

    Xtrain = X[train_idx, :, :, :]
    ytrain = y[train_idx]
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, X.shape[3], 1)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    if require_test == True:
        Xtest = np.delete(X, train_idx, 0)
        ytest = np.delete(y, train_idx, 0)
        Xtest = Xtest.reshape(-1, patch_size, patch_size, X.shape[3], 1)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2)
        return Xtrain, ytrain, Xtest, ytest
    else:
        return Xtrain, ytrain


def data_pre_all(X, y, train_idx, require_test=True):

    X, y = createImageCubes(X, y, windowSize=patch_size)

    Xtrain = X[train_idx, :, :, :]
    ytrain = y[train_idx]
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, X.shape[3], 1)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    if require_test == True:
        Xtest = X
        ytest = y
        Xtest = Xtest.reshape(-1, patch_size, patch_size, X.shape[3], 1)
        Xtest = Xtest.transpose(0, 4, 3, 1, 2)
        return Xtrain, ytrain, Xtest, ytest
    else:
        return Xtrain, ytrain
