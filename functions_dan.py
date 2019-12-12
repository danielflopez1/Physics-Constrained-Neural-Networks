import numpy
import numpy as np
import scipy.io as sio
from scipy.sparse import bsr_matrix


# import theano.tensor as T
# import theano

def sequence_input2(inp_data, out_data, t_step, axis, pred_stepdiff):
    ind = np.asarray(inp_data.shape)
    ind[axis] = inp_data.shape[axis] - 1 * t_step + 1 - pred_stepdiff
    ind2 = np.asarray(out_data.shape)
    ind2[axis] = out_data.shape[axis] - 1 * t_step + 1 - pred_stepdiff
    inp_datanew = np.zeros([ind[0], t_step, ind[1], ind[2], ind[3]])
    # out_datanew = np.zeros([ind2[0],t_step,ind2[1]])
    # print(inp_data.shape)
    # print(inp_datanew.shape)
    for i in range(0, inp_datanew.shape[axis]):
        tn = i
        tnplus1 = i + t_step

        inp_datanew[i, :, :, :, :] = inp_data[tn:tnplus1, :, :, :]
    #  out_datanew[i,:,:] = out_data[tnplus1 + pred_stepdiff: tnplus1 + t_step + pred_stepdiff,:]
    #  print(i)
    return inp_datanew

def sequence_input3(inp_data, t_step, axis, pred_stepdiff):
    ind = np.asarray(inp_data.shape)
    #print(ind,inp_data.shape[axis],  2 * t_step + 1, pred_stepdiff)
    ind[axis] = inp_data.shape[axis] - 2 * t_step + 1 - pred_stepdiff
    if(ind[axis]<1):
        ind[axis] = 1
    #print(ind)
    inp_datanew = np.zeros([ind[0], t_step, ind[1], ind[2], ind[3]])

    for i in range(0, inp_datanew.shape[axis]):
        tn = i
        tnplus1 = i + t_step

        inp_datanew[i, :, :, :, :] = inp_data[tn:tnplus1, :, :, :]
    #  print(i)
    return inp_datanew

def sequence_input(inp_data, out_data, t_step, axis, pred_stepdiff):
    ind = np.asarray(inp_data.shape)
    ind[axis] = inp_data.shape[axis] - 2 * t_step + 1 - pred_stepdiff
    ind2 = np.asarray(out_data.shape)
    ind2[axis] = out_data.shape[axis] - 2 * t_step + 1 - pred_stepdiff
    inp_datanew = np.zeros([ind[0], t_step, ind[1], ind[2], ind[3]])
    out_datanew = np.zeros([ind2[0], t_step, ind2[1]])

    for i in range(0, inp_datanew.shape[axis]):
        tn = i
        tnplus1 = i + t_step

        inp_datanew[i, :, :, :, :] = inp_data[tn:tnplus1, :, :, :]
        out_datanew[i, :, :] = out_data[tnplus1 + pred_stepdiff: tnplus1 + t_step + pred_stepdiff, :]
    #  print(i)
    return inp_datanew, out_datanew


def data_normalize_v2(x_data, norm_type, data_mean, data_range, norm_axis):
    #print('Pre-norm Stats')
    #print 'max:', np.amax(x_data)
    #print 'min:', np.amin(x_data)
    #print 'mean:', np.mean(x_data)

    if type(data_mean) == str:
        data_mean = numpy.mean(x_data, axis=norm_axis)

    if type(data_range) == str:
        data_range = numpy.max(x_data, axis=norm_axis) - numpy.min(x_data, axis=norm_axis)
        if norm_type == 'var':
            data_range = numpy.var(x_data, axis=norm_axis)

    if norm_type == 'mean':
        #print(data_mean)
        x_data = (x_data - data_mean) / data_range

    if norm_type == 'var':
        #  data_range = numpy.var(x_data)
        x_data = (x_data - data_mean) / data_range

    if norm_type == 'max':
        #   data_range = numpy.amax(x_data)
        data_mean = 0
        x_data = (x_data - data_mean) / data_range

    #print('Post-norm Stats')
    #print 'max:', np.amax(x_data)
    #print 'min:', np.amin(x_data)
    #print 'mean:', np.mean(x_data)

    return x_data, data_mean, data_range


def data_normalize(x_data, norm_type, data_mean, data_range):
    #print('Pre-norm Stats')
    #print 'max:', np.amax(x_data)
    #print 'min:', np.amin(x_data)
    #print 'mean:', np.mean(x_data)

    if type(data_mean) == str:
        data_mean = numpy.mean(x_data)

    if type(data_range) == str:
        data_range = numpy.max(x_data) - numpy.min(x_data)

    if norm_type == 'mean':
        x_data = (x_data - data_mean) / data_range

    if norm_type == 'STD':
        data_range = numpy.std(x_data)
        x_data = (x_data - data_mean) / data_range

    if norm_type == 'max':
        #   data_range = numpy.amax(x_data)
        data_mean = 0
        x_data = (x_data) / data_range

    #print('Post-norm Stats')
    #print 'max:', np.amax(x_data)
    #print 'min:', np.amin(x_data)
    #print 'mean:', np.mean(x_data)

    return x_data, data_mean, data_range


def data_normalize_vold(x_data, norm_type, data_mean, data_range, norm_axis):
    #print('Pre-norm Stats')
    #print 'max:', np.amax(x_data)
    #print 'min:', np.amin(x_data)
    #print 'mean:', np.mean(x_data)

    if data_mean == 'non':
        data_mean = numpy.mean(x_data)

    if data_range == 'non':
        data_range = numpy.max(x_data) - numpy.min(x_data)

    if norm_type == 'mean':
        x_data = (x_data - data_mean) / data_range

    if norm_type == 'STD':
        data_range = numpy.std(x_data)
        x_data = (x_data - data_mean) / data_range

    if norm_type == 'max':
        #   data_range = numpy.amax(x_data)
        data_mean = 0
        x_data = (x_data - data_mean) / data_range

    #print('Post-norm Stats')
    #print 'max:', np.amax(x_data)
    #print 'min:', np.amin(x_data)
    #print 'mean:', np.mean(x_data)


def data_randomize(x_data, y_data, train_set_ratio, valid_set_ratio):
    m = x_data.shape[0]

    ## Random Index Selector
    rand_select = numpy.random.permutation(m)

    i11 = 0
    i12 = (numpy.around(m * train_set_ratio).astype(int))
    i13 = 0
    i14 = 0  # y_data.shape[1]
    i21 = (numpy.around(m * train_set_ratio).astype(int))
    i22 = ((numpy.around(m * (train_set_ratio + valid_set_ratio)).astype(int)))
    i23 = i13
    i24 = i14
    i31 = (numpy.around(m * (train_set_ratio + valid_set_ratio)).astype(int))
    i32 = m
    i33 = i13
    i34 = i14

    train_set_x = x_data[rand_select[i11:i12], :, :]
    train_set_y = y_data[rand_select[i11:i12], :]
    valid_set_x = x_data[rand_select[i21:i22], :, :]
    valid_set_y = y_data[rand_select[i21:i22], :]
    test_set_x = x_data[rand_select[i31:i32], :, :]
    test_set_y = y_data[rand_select[i31:i32], :]

    # test_set_x = T.cast(theano.shared(test_set_x),'float32')
    # test_set_y = T.cast(theano.shared(test_set_y),'float32')
    # train_set_x = T.cast(theano.shared(train_set_x),'float32')
    # train_set_y = T.cast(theano.shared(train_set_y),'float32')
    # valid_set_x = T.cast(theano.shared(valid_set_x),'float32')
    # valid_set_y = T.cast(theano.shared(valid_set_y),'float32')

    return (train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y)


def get1DCosine(sampleS, trainF):
    inp = np.random.rand(sampleS, 1)
    # randInd=np.random.choice(sampleS,sampleS,replace=False)
    cutoff = np.rint(trainF * sampleS).astype(int)
    inp_train, inp_test = inp[np.arange(cutoff)], inp[np.arange(cutoff, sampleS)]
    out_train = np.cos(2 * np.pi * inp_train)
    out_test = np.cos(2 * np.pi * inp_test)

    return (inp_train, inp_test, out_train, out_test)


def get2DCosine(sampleS, trainF):
    inp = np.random.rand(sampleS, 2)

    # randInd=np.random.choice(sampleS,sampleS,replace=False)
    cutoff = np.rint(trainF * sampleS).astype(int)
    inp_train, inp_test = inp[np.arange(cutoff)], inp[np.arange(cutoff, sampleS)]
    out_train = np.cos(2 * np.pi * np.multiply(inp_train[:, 0], inp_train[:, 1]))
    out_test = np.cos(2 * np.pi * np.multiply(inp_test[:, 0], inp_test[:, 1]))

    return (inp_train, inp_test, out_train, out_test)


def get3DCosine(sampleS, trainF):
    inp = np.random.rand(sampleS, 3)

    # randInd=np.random.choice(sampleS,sampleS,replace=False)
    cutoff = np.rint(trainF * sampleS).astype(int)
    inp_train, inp_test = inp[np.arange(cutoff)], inp[np.arange(cutoff, sampleS)]
    out_train = np.cos(2 * np.pi * np.multiply(inp_train[:, 0], inp_train[:, 1])) + np.cos(
        2 * np.pi * np.multiply(inp_train[:, 1], inp_train[:, 2]))
    out_test = np.cos(2 * np.pi * np.multiply(inp_test[:, 0], inp_test[:, 1])) + np.cos(
        2 * np.pi * np.multiply(inp_test[:, 1], inp_test[:, 2]))

    return (inp_train, inp_test, out_train, out_test)


def arbitraryF(trainF):
    mat_contents = sio.loadmat('data100.mat')
    inp = mat_contents['inputMat']
    inp = inp / np.amax(inp)
    out = mat_contents['out']
    out = out / np.amax(out)
    sampleS = np.size(out[:, 0])
    cutoff = np.rint(trainF * sampleS).astype(int)
    inputL = np.size(inp[0, :])
    inp_train, inp_test = inp[np.arange(cutoff)], inp[np.arange(cutoff, sampleS)]
    out_train, out_test = out[np.arange(cutoff)], out[np.arange(cutoff, sampleS)]

    return (inp_train, inp_test, out_train, out_test, inputL)


def data_extract(trainF, file1, file2, file3, file4, file5, file6, file7):
    mat1 = sio.loadmat(file1)

    mat2 = sio.loadmat(file2)
    inp_set1 = mat2['inp']
    out_set1 = mat2['out']
    mat3 = sio.loadmat(file3)
    inp_set2 = mat3['inp']
    out_set2 = mat3['out']

    mat4 = sio.loadmat(file4)
    inp_set5 = mat4['inp']
    out_set5 = mat4['out']

    mat5 = sio.loadmat(file5)
    inp_set6 = mat5['dataset_new_inp']
    out_set6 = mat5['dataset_new_out']

    mat6 = sio.loadmat(file6)
    inp_set7 = mat6['inp']
    out_set7 = mat6['out']

    mat7 = sio.loadmat(file7)
    inp_set8 = mat7['inp']
    out_set8 = mat7['out']

    inp = mat1['inp']
    # inp_set1=inp_set1
    # inp_set2=inp_set2
    out = mat1['out']
    # print(out.shape)
    # out=out/np.amax(out)
    N_features = np.size(inp[0, :, 0])
    depth = np.size(inp[0, 0, :])
    sampleS = np.size(out[:, 0])
    cutoff = np.rint(trainF * sampleS).astype(int)
    inputL = np.size(inp[0, :])
    inp_set3, inp_set4 = inp[np.arange(cutoff)], inp[np.arange(cutoff, sampleS)]
    out_set3, out_set4 = out[np.arange(cutoff)], out[np.arange(cutoff, sampleS)]

    return (
    inp_set1, out_set1, inp_set2, out_set2, inp_set3, out_set3, inp_set4, out_set4, inp_set5, out_set5, inp_set6,
    out_set6, inp_set7, out_set7, inp_set8, out_set8, N_features, depth)
