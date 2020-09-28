#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <iostream>
#include "ReadMNIST.h"

using namespace std;
using namespace cv;

HOGDescriptor hog(
    Size(28, 28),         //winSize or imagesize
    Size(14, 14),         //blocksize used in vector normalization,
    Size(7, 7),           //blockStride or stepsize to next cell,
    Size(7, 7),           //cellSize,
    18,                   //number of bins in the histogram,
    1,                    //derivAper,
    -1,                   //winSigma,
    HOGDescriptor::L2Hys, //histogramNormType,
    0.2,                  //L2HysThresh,
    0,                    //gammal correction,
    64,                   //nlevels=64
    1);

static void help()
{
    cout
        << "\n"
        << "This program uses Histogram of Oriented Gradients (HOG) to create an image characteristic"
        << "and a Support Vector Machine (SVM) model to classify digits of the MNIST database. "
        << "The code is inspired by https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/. "
        << "In case one wants to experiment with the algorithm, it is advised to vary the first 5 parameters in the HOGDescriptor and the C & gamma parameters of the SVM model. "
        << "For information on the subject, visit the above link or opencv.org. \n"
        << "\n"
        << "Usage:\n"
        << "\t- ./app or ./app -train           To train and save SVM results\n"
        << "\t- ./app -load                     To load results from previously trained classifier\n"
        << endl;
}

Mat deskew(const Mat &img)
{
    // Compute moments to deskew/rotate the image
    Moments m = moments(img);
    if (abs(m.mu02) < 1e-2)
    {
        // If moment is small, no deskewing is needed.
        return img.clone();
    }
    // Calculate rotation based on central momemts.
    double skew = m.mu11 / m.mu02;
    // Calculate affine transform to correct rotation.
    Mat warpMat = (Mat_<double>(2, 3) << 1, skew, -0.5 * 28 * skew, 0, 1, 0);

    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(), WARP_INVERSE_MAP | INTER_LINEAR);

    return imgOut;
}

vector<Mat> createDeskewedData(const vector<Mat> &imSet)
{
    vector<Mat> imDeskewedSet;
    for (auto &im : imSet)
    {
        Mat img = im;
        imDeskewedSet.push_back(deskew(img));
    }
    return imDeskewedSet;
}

vector<vector<float>> createHOGData(const vector<Mat> &imSet)
{
    vector<vector<float>> hogSet;
    vector<float> descriptor;
    for (auto &im : imSet)
    {
        hog.compute(im, descriptor);
        hogSet.push_back(descriptor);
    }

    return hogSet;
}

Mat convertVectorToMat(const vector<vector<float>> &vec)
{
    Mat mat(vec.size(), vec[0].size(), CV_32FC1);
    int nRows = vec.size();
    int nCols = vec[0].size();
    for (int r = 0; r < nRows; ++r)
    {
        for (int c = 0; c < nCols; ++c)
        {
            mat.at<float>(r, c) = vec[r][c];
        }
    }
    return mat;
}

float SVMevaluate( Mat testResponse, const vector<int> &testLabels, vector<int> &testNegative)
{
    // Determine accuracy of classifier and retrieve indices of incorrectly classified images
    Mat pix;
    float count, accuracy;
    for (int i = 0; i < testResponse.rows; ++i)
    {
        if (testResponse.at<float>(i, 0) == testLabels[i])
            count = count + 1;
        else
            testNegative.push_back(i);
    }
    accuracy = (count / testResponse.rows) * 100;
    return accuracy;
}

void getSVMParams(Ptr<ml::SVM> svm)
{
    cout << "Kernel type        : " << svm->getKernelType() << endl;
    cout << "Type               : " << svm->getType() << endl;
    cout << "C                  : " << svm->getC() << endl;
    cout << "Degree             : " << svm->getDegree() << endl;
    cout << "Nu                 : " << svm->getNu() << endl;
    cout << "Gamma              : " << svm->getGamma() << endl;
}

int display(const vector<Mat> &imSet, Mat answer, string nameWindow)
{
    // Displays digit images with the classifier responds added below
    namedWindow(nameWindow, WINDOW_AUTOSIZE);
    Mat im;
    Mat imAnsSet;
    float *ptr;
    int num;
    for (int i = 0; i<imSet.size(); ++i)
    {
        // Create image to write classification answer to
        Mat imAns(84, 84, CV_8UC1, Scalar(0, 0, 0));

        // Obtain classifier answer and write on image
        ptr = answer.ptr<float>(i);
        num = (int) *ptr;
        cv::putText(imAns,
                    to_string(num),                 // Text
                    cv::Point(42, 42),              // Coordinates
                    cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                    1.5,                            // Scale. 1.5 = 1.5x bigger
                    cv::Scalar(255),                // Color (BGR)
                    1);

        if (i > 0)
        {
            hconcat(imAnsSet, imAns, imAnsSet);
            hconcat(im, imSet[i], im);
        }
        else
        {
            imAnsSet = imAns;
            im = imSet[i];
        }
    }
    // Resize to proper readible size
    cv::resize(im, im, cv::Size(), 3, 3);
    // Display answers below the image
    vconcat(im, imAnsSet, im);

    // Write instructions on the image
    cv::putText(im,
                "Press <q> to quit / any key for next",            // Text
                cv::Point(12, 156),             // Coordinates
                cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                0.5,                            // Scale. 1.5 = 1.5x bigger
                cv::Scalar(255),                // Color (BGR)
                1);

    imshow(nameWindow, im);
    int key = waitKey(0);
    destroyWindow(nameWindow);
    return key;
}

void showResults(const vector<Mat> &imSet,  Mat ansSet, string nameWindow)
{
    // Creates subsets of 5 images to display
    auto it = imSet.begin();
    int iAns = 0;
    int key;
    while (key != 'q')
    {
        int numOfImages = 5;
        // In case less than 5 images left, determine a new numOfImages
        if (it + numOfImages > imSet.end()){
            while(it+numOfImages>imSet.end()){  
                numOfImages--; 
            }
        }
        // Extract subset to display
        vector<Mat> imSubSet(it, it + numOfImages);
        Mat ansSubSet = ansSet.rowRange(iAns, iAns + numOfImages);
        key = display(imSubSet, ansSubSet, nameWindow);

        iAns += numOfImages;
        it += numOfImages;
        if( it == imSet.end()) break;
    }
}

int main(int argc, char *argv[])
{
    help();

    // Used to compute the time to load data
    double loadTime = (double)getTickCount();

    // Set paths to files
    std::string const setPathTrainingImages = "Training_images/train-images.idx3-ubyte";
    std::string const setPathTrainingLabels = "Training_images/train-labels.idx1-ubyte";
    std::string const setPathTestImages = "Test_images/t10k-images.idx3-ubyte";
    std::string const setPathTestLabels = "Test_images/t10k-labels.idx1-ubyte";

    // Read the MNIST database
    cout << "Loading the training dataset..." << endl;
    vector<Mat> const imTraining = readMnistImages(setPathTrainingImages);
    vector<int> const labelTraining = readMnistLabels(setPathTrainingLabels, imTraining.size());
    cout << "Loading the test dataset..." << endl;
    vector<Mat> const imTest = readMnistImages(setPathTestImages);
    vector<int> const labelTest = readMnistLabels(setPathTestLabels, imTest.size());

    // Create deskewed images
    cout << "Deskew images if necessary..." << endl;
    vector<Mat> imDeskewedTraining = createDeskewedData(imTraining);
    vector<Mat> imDeskewedTest = createDeskewedData(imTest);

    // Generate HOG data
    cout << "Generate image descriptor vectors..." << endl;
    vector<vector<float>> trainHOG = createHOGData(imDeskewedTraining);
    vector<vector<float>> testHOG = createHOGData(imDeskewedTest);
    cout << "Descriptor Size    : " << trainHOG[0].size() << endl;

    // Convert the HOG vector of vectors into a training matrix
    Mat trainMat = convertVectorToMat(trainHOG);
    Mat testMat = convertVectorToMat(testHOG);

    // Compute loading duration
    loadTime = ((double)getTickCount() - loadTime) / getTickFrequency();
    cout << "Time required to load the dataset: " << loadTime << " seconds" << endl;

    // Declare SVM model before either training or loading
    Ptr<ml::SVM> svm;

    if (argc == 1 || string(argv[1]) == "-train")
    {
        // Used to compute the time to load data
        double learnTime = (double)getTickCount();

        // Set SVM type and parameters
        svm = ml::SVM::create();
        svm->setType(ml::SVM::C_SVC);
        // Set SVM Kernel to Radial Basis Function (RBF)
        svm->setKernel(ml::SVM::RBF); 
        svm->setC(10);
        svm->setGamma(.50625);//.50625

        ///////////////////////
        // Train the SVM model
        Ptr<ml::TrainData> td = ml::TrainData::create(trainMat, ml::ROW_SAMPLE, labelTraining);
        svm->train(td);
        // svm->trainAuto(td);
        svm->save("Results/eyeGlassClassifierModel.yml");

        // Compute SVM computation duration
        learnTime = ((double)getTickCount() - learnTime) / getTickFrequency();

        cout << "Learning runtime(s): " << learnTime << endl;
        
    }

    if (string(argv[1]) == "-load")
    {
        svm = Algorithm::load<ml::SVM>("Results/eyeGlassClassifierModel.yml");
    }

    //////////////////////////
    // Apply model to test set
    Mat testResponse;
    svm->predict(testMat, testResponse);

    ////////////// Find Accuracy ///////////
    vector<int> testNegative;
    getSVMParams(svm);
    float accuracy = SVMevaluate(testResponse, labelTest, testNegative);
    cout << "Accuracy           : " << accuracy << endl;
    showResults(imTest, testResponse, "Test results");

    /// Collect & show results of incorrectly classified digits
    vector<Mat> imNegative;
    Mat negativeResponse(testNegative.size(), 1, testResponse.type());
    int count = 0;
    for (int &sample : testNegative)
    {
        imNegative.push_back(imTest[sample]);
        negativeResponse.at<float>(count, 0) = testResponse.at<float>(sample, 0);
        count++;
    }

    showResults(imNegative, negativeResponse, "Incorrectly classified");

    return 0;
}