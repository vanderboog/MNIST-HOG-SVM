# MNIST: handwritten digit classification (HOG + SVM)

This program uses Histogram of Oriented Gradients (HOG) to create an image characteristic and a Support Vector Machine (SVM) model to classify digits of the MNIST database. The code is inspired by https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/, however uses the MNIST database allowing to obtain an accuracy of 99.41% (with approximately 3 minutes of training).

In case one wants to experiment with the algorithm, it is advised to vary the second to fifth parameters in the HOGDescriptor and the C & gamma parameters of the SVM model. For information on the subject, visit the above link or https://opencv.org.

Usage example to compile & run:

**`g++ -std=c++17 *.cpp -o app pkg-config --cflags --libs opencv4 && ./app -train`**

When the training/loading is finished the test results are displayed. On exit, incorrectly classified samples are shown. Both times, press \<q> to exit or any other key to visualize the next result.

Packages used:

- OpenCV

| Usage arguments |                                              |
| --------------- | -------------------------------------------- |
| -train          | Train and save a SVM model                   |
| -load           | Load a previously trained model from storage |



