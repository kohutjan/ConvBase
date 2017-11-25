#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <iomanip>
#include "net.hpp"
#include <math.h>

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
  return image;
}

Tensor4D ConvertImageToTensor4D(Mat image, float mean, float scale)
{
  int nChannels = image.channels();
  int nRows = image.rows;
  int nCols = image.cols;
  Tensor4D tensor = Tensor4D(1, nChannels, nRows, nCols);
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

vector<float> SoftMax(Tensor4D tensor)
{
  float * data = tensor.GetData();
  vector<float> tensorExp(tensor.GetSize());
  for (size_t i = 0; i < tensorExp.size(); ++i)
  {
    tensorExp[i] = exp(data[i]);
  }
  float sumExp = 0.0;
  for (auto& val: tensorExp)
  {
    sumExp += val;
  }
  vector<float> tensorSoftMax(tensor.GetSize());
  for (size_t i = 0; i < tensorSoftMax.size(); ++i)
  {
    tensorSoftMax[i] = tensorExp[i] / sumExp;
  }
  return tensorSoftMax;
}

void PrintScore(Tensor4D tensor)
{
  vector<float> tensorSoftMax = SoftMax(tensor);
  cout << setprecision(2);
  for (size_t i = 0; i < tensorSoftMax.size(); ++i)
  {
    cout << tensorSoftMax[i] << " ";
  }
  cout << endl;
}

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = LoadImage(argv[1]);

    if(! image.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Net net;
    net.Load("../nets/2xC32_P2_2xC32_P2_2xC32_P2_FC256_FC10_iter_3000.convbase");
    net.Init();
    net.AddTensor4DToContainer(net.inputs.begin()->first, ConvertImageToTensor4D(image, 127.0, 0.007840157));
    net.Forward();
    string topName = net.operators.back()->GetBottomName()[0];
    Tensor4D output = net.GetTensor4DFromContainer(topName);
    PrintScore(output);

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
