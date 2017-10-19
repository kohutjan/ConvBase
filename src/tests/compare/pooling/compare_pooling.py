import numpy as np
import sys
import argparse
import os
import math
import random
import subprocess
from google.protobuf.text_format import Merge
import tempfile


def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="Compare pooling. \
            Error is mean(abs(DifferenceOfOutputs))/mean(abs(ExpectedOutput)).")

    parser.add_argument('-tn', '--tests-number',
                        type=int,
                        default=1,
                        help="Number of tests (default 1)")
    parser.add_argument('-e', '--convbase-executable',
                        required=True,
                        help="compute_convolution exacutable")
    parser.add_argument('-f', '--forward',
                        action='store_true',
                        help='Compare forward pass.')
    parser.add_argument('-b', '--backward',
                        action='store_true',
                        help='Compare backward pass.')
    parser.add_argument('-p', '--params',
                        help="Syntax: Input shape: N,H,W,C  Pooling params: kernelSize,stride")
    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        help='Set cpu mode (default gpu mode)')
    parser.add_argument('-v', '--verbose',
                        action='store_true')

    args = parser.parse_args()

    if not (args.forward or args.backward):
        sys.stdout.write("No tests to be done.\n")
        sys.exit(1)

    return args


def main():
    args = parse_args()

    if not args.verbose:
        os.environ['GLOG_minloglevel'] = '2'

    import caffe
    from caffe.proto import caffe_pb2

    if args.cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()

    if args.verbose:
        if args.cpu:
            print("CPU mode set.")
        else:
            print("GPU mode set.")

    errorSum = np.zeros(2)

    for i in range(args.tests_number):
        tmpNetProto = tempfile.NamedTemporaryFile()
        tmpNetProto.write(createPoolingNet(args.params))
        tmpNetProto.flush()
        net = caffe.Net(tmpNetProto.name, caffe.TEST)
        deploy = caffe_pb2.NetParameter()
        Merge((open(tmpNetProto.name,'r').read()), deploy)
        tmpNetProto.close()
        sys.stdout.write("{}. ".format(i + 1))
        if not args.verbose:
            sys.stdout.write("Input shape: {}, {},{},{} ".format(net.blobs['data'].data.shape[0],
                                                                 net.blobs['data'].data.shape[2],
                                                                 net.blobs['data'].data.shape[3],
                                                                 net.blobs['data'].data.shape[1]))
            poolingParams = deploy.layer[1].pooling_param
            sys.stdout.write("Pooling params: {},{} ".format(poolingParams.kernel_size,
                                                             poolingParams.stride))
            sys.stdout.write("Output shape: {},{},{}".format(net.blobs['pooling'].data.shape[2],
                                                                net.blobs['pooling'].data.shape[3],
                                                                net.blobs['pooling'].data.shape[1]))

        net.blobs['data'].data[...] = np.random.random_sample(net.blobs['data'].data.shape) - 0.5
        net.blobs['pooling'].diff[...] = np.random.random_sample(net.blobs['pooling'].diff.shape) - 0.5


        errorSum += comparePooling(net, deploy, args.forward, args.backward,
                                   args.convbase_executable,
                                   outputPreffix="pooling_",
                                   verbose=args.verbose)

    meanError = errorSum / args.tests_number

    print ("\n#############################################################")
    print ("Number of tests: {}\n".format(args.tests_number))
    if args.forward:
        print ("Mean forward error: {}".format(meanError[0]))
    if args.backward:
        print ("Mean backward error: {}".format(meanError[1]))
    print ("#############################################################")


def comparePooling(net, deploy, forward, backward, convbaseExecutable, bottomName='data',
                   topName='pooling', outputPreffix="", verbose=False):

    if verbose:
        stdOut = sys.stdout
    else:
        stdOut = open(os.devnull, 'w')

    # Prepare input for C++ implementation
    if forward:
        try:
            os.remove(outputPreffix + "data_input.txt")
        except OSError:
            pass
        dataInputFile = open(outputPreffix + "data_input.txt", "w")
        nhwcDataInput = np.swapaxes(np.swapaxes(net.blobs[bottomName].data, 1, 2), 2, 3).reshape(-1)
        dataInputFile.write("{} {} {} {}\n".format(net.blobs[bottomName].data.shape[0],
                                                   net.blobs[bottomName].data.shape[2],
                                                   net.blobs[bottomName].data.shape[3],
                                                   net.blobs[bottomName].data.shape[1]))
        for value in nhwcDataInput:
            dataInputFile.write("{}\n".format(value))
        dataInputFile.close()

    if backward:
        try:
            os.remove(outputPreffix + "gradients_input.txt")
        except OSError:
            pass
        gradientsInputFile = open(outputPreffix + "gradients_input.txt", "w")
        nhwcGradientsInput = np.swapaxes(np.swapaxes(net.blobs[topName].diff, 1, 2), 2, 3).reshape(-1)
        gradientsInputFile.write("{} {} {} {}\n".format(net.blobs[topName].diff.shape[0],
                                                        net.blobs[topName].diff.shape[2],
                                                        net.blobs[topName].diff.shape[3],
                                                        net.blobs[topName].diff.shape[1]))
        for value in nhwcGradientsInput:
            gradientsInputFile.write("{}\n".format(value))
        gradientsInputFile.close()


    try:
        os.remove(outputPreffix + "params.txt")
    except OSError:
        pass
    paramsFile = open(outputPreffix + "params.txt", "w")
    paramsFile.write("Input\n")
    paramsFile.write("1\n")
    paramsFile.write("bottom {} {} {} {}\n".format(net.blobs[bottomName].data.shape[0],
                                                  net.blobs[bottomName].data.shape[2],
                                                  net.blobs[bottomName].data.shape[3],
                                                  net.blobs[bottomName].data.shape[1]))
    paramsFile.write("\nPooling\n")
    paramsFile.write("1 bottom 1 top\n")
    params = deploy.layer[1]
    paramsFile.write("{} {}\n".format(params.pooling_param.kernel_size,
                                      params.pooling_param.stride))
    paramsFile.close()

    net.forward()
    net.backward()

    if forward:
        convbaseArgs = [convbaseExecutable,
                        outputPreffix + "params.txt",
                        outputPreffix + "data_input.txt",
                        outputPreffix + "forward_convbase_output.txt",
                        "forward"]
        convbaseForward = subprocess.Popen(convbaseArgs, stdout=stdOut)
        convbaseForward.wait()

    if backward:
        convbaseArgs = [convbaseExecutable,
                        outputPreffix + "params.txt",
                        outputPreffix,
                        outputPreffix,
                        "bf"]
        convbaseBackward = subprocess.Popen(convbaseArgs, stdout=stdOut)
        convbaseBackward.wait()

    forwardError = 0
    backwardError = 0

    sys.stdout.write("\n")

    if forward:
        try:
            os.remove(outputPreffix + "forward_caffe_output.txt")
        except OSError:
            pass
        outputFile = open(outputPreffix + "forward_caffe_output.txt", "w")
        nhwcOutput = np.swapaxes(np.swapaxes(net.blobs[topName].data, 1, 2), 2, 3).reshape(-1)
        for value in nhwcOutput:
            outputFile.write("{}\n".format(value))
        outputFile.close()

        error, code = compareOutputs(outputPreffix + "forward_caffe_output.txt",
                                     outputPreffix + "forward_convbase_output.txt")
        sys.stdout.write("Forward ")
        forwardError = printError(verbose, error, code)

    if backward:
        try:
            os.remove(outputPreffix + "backward_caffe_output.txt")
        except OSError:
            pass
        outputFile = open(outputPreffix + "backward_caffe_output.txt", "w")
        nhwcOutput = np.swapaxes(np.swapaxes(net.blobs[bottomName].diff, 1, 2), 2, 3).reshape(-1)
        for value in nhwcOutput:
            outputFile.write("{}\n".format(value))
        outputFile.close()

        error, code = compareOutputs(outputPreffix + "backward_caffe_output.txt",
                                     outputPreffix + "backward_convbase_output.txt")
        sys.stdout.write("Backward ")
        backwardError = printError(verbose, error, code)

    return np.asarray((forwardError, backwardError))


def printError(verbose, error, code):

    sys.stdout.write("error: ")
    if code == 'OK':
        sys.stdout.write("{}\n".format(error))
        return error
    elif code == 'DO':
        sys.stdout.write("Different outputs size.\n")
    elif code == 'CC':
        sys.stdout.write("Corrupted caffe output.\n")
    elif code == 'CX':
        sys.stdout.write("Corrupted ConvBase output.\n")

    return 0.0


def compareOutputs(caffeOutputFile, convbaseOutputFile):
    caffeOutput = []
    convbaseOutput = []
    with open(caffeOutputFile) as f:
        caffeOutput = f.read().splitlines()
    with open(convbaseOutputFile) as f:
        convbaseOutput = f.read().splitlines()

    try:
        caffeOutput = np.array(map(float, caffeOutput))
    except:
        caffeOutput = []
        pass
    if caffeOutput[np.where(abs(caffeOutput) > 10e+10)].size > 0 or caffeOutput.size == 0:
        return 0.0, 'CC'
    try:
        convbaseOutput = np.array(map(float, convbaseOutput))
    except:
        convbaseOutput = []
        pass
    if convbaseOutput[np.where(abs(convbaseOutput) > 10e+10)].size > 0 or convbaseOutput.size == 0:
        return 0.0, 'CX'

    if caffeOutput.size != convbaseOutput.size:
        return 0.0, 'DO'

    return np.abs(caffeOutput - convbaseOutput).mean()/np.abs(caffeOutput).mean(), 'OK'


def createPoolingNet(params):
    import caffe
    if params is not None:
        params = params.split()
        inputParams = params[2].split(",")
        num = int(inputParams[0])
        height = int(inputParams[1])
        width = int(inputParams[2])
        channels = int(inputParams[3])
        poolingParams = params[5].split(",")
        kernelSize = int(poolingParams[0])
        stride = int(poolingParams[1])
    else:
        kernelSize = random.randint(1, 5)
        stride = random.randint(1, 10)
        num = random.randint(1, 32)
        channels = random.randint(1, 32)
        height = random.randint(1, 32)
        width = random.randint(1, 32)

        # Adjust input to exactly fit conv params
        height = adjustDimension(height, kernelSize, stride, 0)
        width = adjustDimension(width, kernelSize, stride, 0)

    """
    num = 2
    height = 4
    width = 4
    channels = 1
    kernelSize = 2
    stride = 2
    """

    net = caffe.NetSpec()
    net.data = caffe.layers.Input(shape=dict(dim=[num, channels, height, width]))
    net.pooling = caffe.layers.Pooling(net.data, kernel_size=kernelSize, stride=stride, pad=0)

    return "force_backward: true\n" + str(net.to_proto())


def adjustDimension(dimension, kernelSize, stride, pad):
    spaceToMoveKernelDimensionWise = dimension + 2 * pad - kernelSize

    if spaceToMoveKernelDimensionWise < 0:
        dimension -= spaceToMoveKernelDimensionWise
    else:
        dimensionOffset = spaceToMoveKernelDimensionWise % stride
        if (dimensionOffset) != 0:
            dimension += stride - dimensionOffset

    return dimension


if __name__ == "__main__":
    main()
