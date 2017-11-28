#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <iomanip>
#include "net.hpp"

using namespace cv;
using namespace std;

Mat LoadImage(string imageName)
{
  Mat image;
  image = imread(imageName, CV_LOAD_IMAGE_COLOR);
  if (image.rows != 32 or image.cols != 32)
  {
    if (image.rows > 32 or image.cols > 32)
    {
      resize(image, image, Size(32, 32), 0, 0, CV_INTER_AREA);
    }
    else
    {
      resize(image, image, Size(32, 32), 0, 0, CV_INTER_CUBIC);
    }
  }
  if(!image.data)
  {
    cout << "Could not open or find the image." << endl;
    return image;
  }
  return image;
}

Tensor4D ConvertImageToTensor4D(Mat image, float mean, float scale)
{
  int nChannels = image.channels();
  int nRows = image.rows;
  int nCols = image.cols;
  Tensor4D tensor = Tensor4D(1, nRows, nCols, nChannels);
  float * data = tensor.GetData();
  for(int i = 0; i < nRows; ++i)
  {
    uint8_t * row = image.ptr<uint8_t>(i);
    for (int j = 0; j < nCols; ++j)
    {
      for (int k = 0; k < nChannels; ++k)
      {
        float val = row[j * nChannels + nChannels - k - 1];
        val -= mean;
        val *= scale;
        data[i * nCols * nChannels + j * nChannels + k] = val;
      }
    }
  }
  return tensor;
}

void PrintScore(Tensor4D tensor)
{
  float * data = tensor.GetData();
  cout << setprecision(2);
  int maxIndex = 0;
  float maxVal = data[0];
  for (size_t i = 1; i < tensor.GetSize(); ++i)
  {
    if (data[i] > maxVal)
    {
      maxIndex = i;
      maxVal = data[i];
    }
  }

  cout << " is ";
  switch(maxIndex)
  {
    case 0:
      cout << "AIRPLANE";
      break;
    case 1:
      cout << "AUTOMOBILE";
      break;
    case 2:
      cout << "BIRD";
      break;
    case 3:
      cout << "CAT";
      break;
    case 4:
      cout << "DEER";
      break;
    case 5:
      cout << "DOG";
      break;
    case 6:
      cout << "FROG";
      break;
    case 7:
      cout << "HORSE";
      break;
    case 8:
      cout << "SHIP";
      break;
    case 9:
      cout << "TRUCK";
      break;
    default:
      break;
  }
  cout << endl;
  for (size_t i = 0; i < tensor.GetSize(); ++i)
  {
    cout << data[i] << " ";
  }
  cout << endl;
}

void ClassifyImage(Net &net, Mat image)
{
  net.AddTensor4DToContainer(net.inputs.begin()->first, ConvertImageToTensor4D(image, 127.0, 0.007874016));
  net.Forward();
  SoftmaxCrossEntropy * softmax = dynamic_cast<SoftmaxCrossEntropy*>(net.operators.back().get());
  Tensor4D output = softmax->GetSoftmaxTop();
  PrintScore(output);
}

int main( int argc, char** argv )
{
    Net net;
    net.Load(argv[1]);
    vector<int> inputShape = net.inputs.begin()->second;
    if (inputShape[0] != 1)
    {
      inputShape[0] = 1;
      net.inputs.begin()->second = inputShape;
    }
    net.Init();

    cout << "File with paths to classify: " << argv[2] << endl;
    cout << endl;
    cout << "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck" << endl;
    cout << "#############################################################" << endl;
    ifstream fileStream(argv[2]);
    if (fileStream.is_open())
    {
      string filePath;
      while (getline(fileStream, filePath))
      {
        cout << filePath;
        Mat image;
        image = LoadImage(filePath);
        ClassifyImage(net, image);
      }
      fileStream.close();
      cout << "#############################################################" << endl;
      cout << endl;
      cout << endl;
    }
    else
    {
      cerr << "Unable to open file with paths." << endl;
      return false;
    }

    return 0;
}
