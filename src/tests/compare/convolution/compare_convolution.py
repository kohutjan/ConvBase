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

    parser = argparse.ArgumentParser(epilog="Compare convolution. \
            Error is mean(abs(DifferenceOfOutputs))/mean(abs(ExpectedOutput)).")

    parser.add_argument('-tn', '--tests-number',
                        type=int,
                        default=1,
                        help="Number of tests (default 1)")
    parser.add_argument('-e', '--convbase-executable',
                        required=True,
                        help="compute_convolution exacutable")
    parser.add_argument('-s', '--bias',
                        action='store_true',
                        help='Use bias.')
    parser.add_argument('-f', '--forward',
                        action='store_true',
                        help='Compare forward pass.')
    parser.add_argument('-b', '--backward',
                        action='store_true',
                        help='Compare backward pass.')
    parser.add_argument('-w', '--weights',
                        action='store_true',
                        help='Compare weights gradients.')
    parser.add_argument('-p', '--params',
                        help='Syntax: Input shape: N,H,W,C  Conv params: kernelNumber,kernelSize,stride,pad')
    parser.add_argument('-c', '--cpu',
                        action='store_true',
                        help='Set cpu mode (default gpu mode)')
    parser.add_argument('-v', '--verbose',
                        action='store_true')

    args = parser.parse_args()

    if not (args.forward or args.backward or args.weights):
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

    errorSum = np.zeros(3)

    for i in range(args.tests_number):
        tmpNetProto = tempfile.NamedTemporaryFile()
        tmpNetProto.write(createConvolutionNet(args.params, args.bias))
        tmpNetProto.flush()
        net = caffe.Net(tmpNetProto.name, caffe.TEST)
        deploy = caffe_pb2.NetParameter()
        Merge((open(tmpNetProto.name,'r').read()), deploy)
        tmpNetProto.close()
        sys.stdout.write("{}. ".format(i + 1))
        if not args.verbose:
            sys.stdout.write("Bottom shape: {},{},{},{} ".format(net.blobs['data'].data.shape[0],
                                                                 net.blobs['data'].data.shape[2],
                                                                 net.blobs['data'].data.shape[3],
                                                                 net.blobs['data'].data.shape[1]))
            convParams = deploy.layer[1].convolution_param
            sys.stdout.write("Conv params: {},{},{},{} ".format(convParams.num_output,
                                                                convParams.kernel_size[0],
                                                                convParams.stride[0],
                                                                convParams.pad[0]))
            sys.stdout.write("Top shape: {},{},{}".format(net.blobs['convolution'].data.shape[2],
                                                             net.blobs['convolution'].data.shape[3],
                                                             net.blobs['convolution'].data.shape[1]))

        net.blobs['data'].data[...] = np.random.random_sample(net.blobs['data'].data.shape) - 0.5
        net.blobs['convolution'].diff[...] = np.random.random_sample(net.blobs['convolution'].diff.shape) - 0.5
        net.params['convolution'][0].data[...] = np.random.random_sample(net.params['convolution'][0].data.shape) - 0.5
        if args.bias:
            net.params['convolution'][1].data[...] = np.random.random_sample(net.params['convolution'][1].data.shape) * 0.25 - 0.125

        errorSum += compareConvolution(net, deploy, args.forward, args.backward,
                                       args.weights, args.convbase_executable,
                                       outputPreffix="convolution_",
                                       verbose=args.verbose)

    meanError = errorSum / args.tests_number

    print ("\n#############################################################")
    print ("Number of tests: {}\n".format(args.tests_number))
    if args.forward:
        print ("Mean forward error: {}".format(meanError[0]))
    if args.backward:
        print ("Mean backward error: {}".format(meanError[1]))
    if args.weights:
        print ("Mean weights gradients error: {}".format(meanError[2]))
    print ("#############################################################")


def compareConvolution(net, deploy, forward, backward, weightsGradients, convbaseExecutable, bottomName='data',
                       topName='convolution', outputPreffix="", verbose=False):

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
    paramsFile.write("\nConvolution\n")
    paramsFile.write("1 botoom 1 top\n")
    try:
        biases = net.params[deploy.layer[1].name][1].data
        biasTerm = 1
    except:
        biases = None
        biasTerm = 0
    params = deploy.layer[1]
    paramsFile.write("{} {} {} {} {}\n".format(params.convolution_param.num_output,
                                               params.convolution_param.kernel_size[0],
                                               params.convolution_param.stride[0],
                                               params.convolution_param.pad[0],
                                               biasTerm))
    nhwcWeights = np.swapaxes(np.swapaxes(net.params[deploy.layer[1].name][0].data, 1, 2), 2, 3).reshape(-1)
    paramsFile.write("{}".format(net.params[deploy.layer[1].name][0].data.shape[1]))
    for weight in nhwcWeights:
        paramsFile.write(" {}".format(weight))
    paramsFile.write("\n")
    if biases is not None:
        paramsFile.write("{}".format(biases.size))
        for bias in biases:
            paramsFile.write(" {}".format(bias))
    else:
        paramsFile.write("0")

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
                        outputPreffix + "gradients_input.txt",
                        outputPreffix + "backward_convbase_output.txt",
                        "backward"]
        convbaseForward = subprocess.Popen(convbaseArgs, stdout=stdOut)
        convbaseForward.wait()

    forwardError = 0;
    backwardError = 0;
    weightsGradientsError = 0;

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

    return np.asarray((forwardError, backwardError, weightsGradientsError))


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


def createConvolutionNet(params, bias):
    import caffe
    if params is not None:
        params = params.split()
        inputParams = params[2].split(",")
        num = int(inputParams[0])
        height = int(inputParams[1])
        width = int(inputParams[2])
        channels = int(inputParams[3])
        convParams = params[5].split(",")
        kernelNumber = int(convParams[0])
        kernelSize = int(convParams[1])
        stride = int(convParams[2])
        pad = int(convParams[3])
    else:
        kernelNumber = random.randint(1, 32)
        kernelSize = random.randint(1, 3)
        stride = random.randint(1, 10)
        pad = random.randint(0, kernelSize - 1)

        num = random.randint(1, 16)
        channels = random.randint(1, 32)
        height = random.randint(1, 32)
        width = random.randint(1, 32)

        """
        kernelNumber = 1
        kernelSize = 2
        stride = 1
        pad = 1

        num = 2
        channels = 1
        height = 2
        width = 2
        """

        # Adjust input to exactly fit conv params
        height = adjustDimension(height, kernelSize, stride, pad)
        width = adjustDimension(width, kernelSize, stride, pad)

    net = caffe.NetSpec()
    net.data = caffe.layers.Input(shape=dict(dim=[num, channels, height, width]))
    net.convolution = caffe.layers.Convolution(net.data, bias_term=bias,
                                               num_output=kernelNumber,
                                               kernel_size=kernelSize,
                                               stride=stride, pad=pad)

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
