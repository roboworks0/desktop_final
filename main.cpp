#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QPushButton>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QTcpSocket>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define PORT 8000  // Port number to listen on

std::vector<std::vector<cv::Point>> merge_close_objects(std::vector<std::vector<cv::Point>>& contours, int min_distance) {
    std::vector<std::vector<cv::Point>> merged_contours;

    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
        return cv::contourArea(contour1, false) > cv::contourArea(contour2, false);
    });

    while (!contours.empty()) {
        std::vector<cv::Point> contour = contours[0];
        contours.erase(contours.begin());
        bool merged = false;

        for (size_t i = 0; i < merged_contours.size(); i++) {
            cv::Rect rect1 = cv::boundingRect(contour);
            cv::Rect rect2 = cv::boundingRect(merged_contours[i]);

            double distance = std::sqrt(std::pow(rect1.x - rect2.x, 2) + std::pow(rect1.y - rect2.y, 2));

            if (distance < min_distance) {
                merged_contours[i].insert(merged_contours[i].end(), contour.begin(), contour.end());
                merged = true;
                break;
            }
        }

        if (!merged) {
            merged_contours.push_back(contour);
        }
    }

    return merged_contours;
}


// Function to apply non-maximum suppression
void applyNMS(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, float nmsThreshold)
{
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.0, nmsThreshold, indices);

    std::vector<cv::Rect> selectedBoxes;
    std::vector<float> selectedConfidences;

    for (int idx : indices) {
        selectedBoxes.push_back(boxes[idx]);
        selectedConfidences.push_back(confidences[idx]);
    }

    boxes = selectedBoxes;
    confidences = selectedConfidences;
}

// Function to receive and update the frame
void updateFrame(cv::Mat frame, QLabel* frameLabel)
{
    /*// Load the pre-trained YOLO model
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("/home/barisayyildiz/yolov3.cfg", "/home/barisayyildiz/yolov3.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class labels
    std::vector<std::string> classLabels;
    QFile labelFile("/home/barisayyildiz/coco.names");
    if (!labelFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Failed to open label file: " << labelFile.fileName();
        return;
    }
    QTextStream in(&labelFile);
    while (!in.atEnd()) {
        classLabels.push_back(in.readLine().toStdString());
    }
    labelFile.close();

    // Preprocess frame and create blob
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);

    // Forward pass through the network
    net.setInput(blob);
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    // Parse output and draw bounding boxes
    float confidenceThreshold = 0.5;  // Confidence threshold for filtering detections
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& out : outs) {
        // Get confidence, class ID, and bounding box
        for (int i = 0; i < out.rows; ++i) {
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            if (confidence > confidenceThreshold) {
                int centerX = static_cast<int>(out.at<float>(i, 0) * frame.cols);
                int centerY = static_cast<int>(out.at<float>(i, 1) * frame.rows);
                int width = static_cast<int>(out.at<float>(i, 2) * frame.cols);
                int height = static_cast<int>(out.at<float>(i, 3) * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    // Apply non-maximum suppression to remove overlapping bounding boxes
    applyNMS(boxes, confidences, nmsThreshold);

    // Draw bounding boxes and labels on the frame
    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 0), 2);

        std::string label = classLabels[classIds[i]] + ": " + std::to_string(confidences[i]);
        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::putText(frame, label, cv::Point(boxes[i].x, boxes[i].y - labelSize.height - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    */

    cv::Mat imgGry;
    cv::cvtColor(frame, imgGry, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(imgGry, imgGry, cv::Size(5, 5), 0);

    // Apply adaptive thresholding to handle different lighting conditions
    cv::Mat thrash;
    cv::adaptiveThreshold(imgGry, thrash, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 7);

    // Apply morphological operations for noise removal
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(thrash, thrash, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(thrash, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Merge close objects
    int min_distance = 30;  // Minimum distance threshold to merge objects
    std::vector<std::vector<cv::Point>> merged_contours = merge_close_objects(contours, min_distance);

    for (const auto& contour : merged_contours) {
        double area = cv::contourArea(contour, false);
        if (area > 300 && area < 7000) {  // Minimum area threshold to exclude small contours
            double epsilon = 0.04 * cv::arcLength(contour, true);  // Adjust epsilon value for closer approximation
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, epsilon, true);
            cv::Rect rect = cv::boundingRect(approx);
            int x = rect.x;
            int y = rect.y - 5;

            if (approx.size() == 3) {
                cv::putText(frame, "Triangle", cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
            else if (approx.size() == 4) {
                cv::putText(frame, "Rectangle", cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
            else if (approx.size() <= 6) {
                cv::putText(frame, "Polygon", cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
            else if (approx.size() > 6 && approx.size() < 20) {
                cv::putText(frame, "Polygon", cv::Point(x, y), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }

            // Draw the smoothed contour
            cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, 0, cv::Scalar(0, 0, 0), 2);
        }
    }

    // Convert the frame to QImage
    QImage image(frame.data, frame.cols, frame.rows, QImage::Format_RGB888);
    image = image.rgbSwapped();

    // Set the QImage as the pixmap of the frame label widget
    frameLabel->setPixmap(QPixmap::fromImage(image));
    frameLabel->setScaledContents(true);
}

// Create a QTcpSocket object
QTcpSocket directionSocket;

// Function to send a message to the server
void sendMessage(const QString& message)
{
    // Convert the message to a QByteArray
    QByteArray data = message.toUtf8();

    // Connect to the server
    directionSocket.connectToHost("192.168.137.153", 5001);

    if (directionSocket.waitForConnected())
    {
        // Send the message to the server
        directionSocket.write(data);
        directionSocket.waitForBytesWritten();
    }
    else
    {
        qDebug() << "Failed to connect to the server";
    }
}

int main(int argc, char *argv[])
{
    // ======================== LAYOUT ======================== //
    QApplication app(argc, argv);

    // Create the main window
    QMainWindow window;

    // Create a central widget
    QWidget* centralWidget = new QWidget(&window);
    window.setCentralWidget(centralWidget);

    // Create a layout for the central widget
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);

    // Create a label widget to display the frame
    QLabel* frameLabel = new QLabel(centralWidget);
    layout->addWidget(frameLabel);

    // Create a widget for the buttons
    QWidget* buttonWidget = new QWidget(centralWidget);
    QHBoxLayout* buttonLayout = new QHBoxLayout(buttonWidget);
    layout->addWidget(buttonWidget);

    // Create the buttons
    QPushButton* forwardButton = new QPushButton("Forward", buttonWidget);
    QPushButton* backwardButton = new QPushButton("Backward", buttonWidget);
    QPushButton* stepRightButton = new QPushButton("Step Right", buttonWidget);
    QPushButton* moveRightButton = new QPushButton("Move Right", buttonWidget);
    QPushButton* stepLeftButton = new QPushButton("Step Left", buttonWidget);
    QPushButton* moveLeftButton = new QPushButton("Move Left", buttonWidget);

    // Add the buttons to the layout
    buttonLayout->addWidget(forwardButton);
    buttonLayout->addWidget(backwardButton);
    buttonLayout->addWidget(stepRightButton);
    buttonLayout->addWidget(moveRightButton);
    buttonLayout->addWidget(stepLeftButton);
    buttonLayout->addWidget(moveLeftButton);

    // Connect button signals to event handlers
    QObject::connect(forwardButton, &QPushButton::clicked, [&]() {
        qDebug() << "Forward button clicked";
        sendMessage("CMD_MOVE_FORWARD#10");
    });

    QObject::connect(backwardButton, &QPushButton::clicked, [&]() {
        qDebug() << "Backward button clicked";
        sendMessage("CMD_MOVE_BACKWARD#10");
    });

    QObject::connect(stepRightButton, &QPushButton::clicked, [&]() {
        qDebug() << "Step Right button clicked";
        sendMessage("CMD_MOVE_RIGHT#10");
    });

    QObject::connect(moveRightButton, &QPushButton::clicked, [&]() {
        qDebug() << "Move Right button clicked";
        sendMessage("CMD_TURN_RIGHT#10");
    });

    QObject::connect(stepLeftButton, &QPushButton::clicked, [&]() {
        qDebug() << "Step Left button clicked";
        sendMessage("CMD_MOVE_LEFT#10");
    });

    QObject::connect(moveLeftButton, &QPushButton::clicked, [&]() {
        qDebug() << "Move Left button clicked";
        sendMessage("CMD_TURN_LEFT#10");
    });


    // Set the layout to the central widget
    centralWidget->setLayout(layout);

    // Initialize the server communication and receive the frame
    // Replace this part with your server communication code from server.cpp
    // ======================== END OF LAYOUT ======================== //



    // ======================== SOCKETS ======================== //
    int server_socket, new_socket, valread;
    struct sockaddr_in address{};
    int addrlen = sizeof(address);

    // Create server socket
    if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind the socket to the specified port
    if (bind(server_socket, (struct sockaddr *) &address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_socket, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "test...\n" << std::endl;

    // Accept incoming connection
    if ((new_socket = accept(server_socket, (struct sockaddr *) &address, (socklen_t *) &addrlen)) < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    //std::vector<uchar> buffer;
    int message_size;

    std::cout << "test..." << "\n";
    int counter = 0;
    // ======================== END OF SOCKETS ======================== //


    cv::Mat frame;
    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, [&]() {
        //cap >> frame;
        // Receive the sensor data in tuple
        char sensor_buffer[12];
        if(read(new_socket, sensor_buffer, sizeof(sensor_buffer)) == 0 ) {
            perror("client disconnected");
            return 1;
        }
        // Unpack the received bytes into a tuple
        int value1 = *reinterpret_cast<int*>(sensor_buffer);
        int value2 = *reinterpret_cast<int*>(sensor_buffer + 4);
        int value3 = *reinterpret_cast<int*>(sensor_buffer + 8);

        //std::cout << "value1 : " << value1 << ", value2 : " << value2 << ", value3 : " << value3 << std::endl;

        //std::cout << "debug1\n";
        if (read(new_socket, &message_size, sizeof(message_size)) == 0) {
            perror("Client disconnected");
            return 1;
        }
        //std::cout << "debug2\n";

        //std::cout << "message size : " << message_size << "\n";

        // Resize the buffer to fit the received frame
        //buffer.resize(message_size);
        ssize_t bytes_read;
        ssize_t total_bytes_read = 0;
        std::cout << "message_size : " << message_size << std::endl;

        cv::Mat image;

        char buffer[320*240*3];
        /*int img_size = read(new_socket, buffer, sizeof(buffer));
        if(img_size == 0){
            std::cout << "img_size : " << img_size << std::endl;
            return 1;
        }
        */

        auto start = high_resolution_clock::now();
        while(total_bytes_read < message_size)
        {
            ssize_t bytes_read = read(new_socket, buffer+total_bytes_read, message_size - total_bytes_read);
            if(bytes_read == 0){
                return 1;
            }else if(bytes_read < 0){
                perror("can't read bytes");
                return 1;
            }
            //std::cout << "bytes_read : " << bytes_read << std::endl;
            total_bytes_read += bytes_read;
        }
        std::cout << "end..." << std::endl;
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Time taken by function: "
            << duration.count() * 0.001 << " miliseconds" << endl;
        // write first 10 bytes
        /*for(int i=0; i<10; i++){
            std::cout << buffer[i] << std::endl;
        }*/

        cv::Mat received_image(240,320, CV_8UC3, buffer);


        /*
        while (total_bytes_read < message_size)
        {
            //std::cout << "total_bytes_read : " << total_bytes_read << "\n";
            ssize_t bytes_read = read(new_socket, buffer.data() + total_bytes_read, message_size - total_bytes_read);
            //std::cout << "bytes_read : " << bytes_read << "\n";
            if (bytes_read == 0)
            {
                // Connection closed by the sender
                break;
            }
            else if (bytes_read < 0)
            {
                // Error reading the byte data
                std::cerr << "Failed to read the byte data." << std::endl;
                break;
            }

            total_bytes_read += bytes_read;
        }
        std::cout << "end..." << std::endl;
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Time taken by function: "
            << duration.count() * 0.001 << " miliseconds" << endl;

        */

        /*
        std::cout << "bytes_read : " << bytes_read << "\n";
        for(int i=0; i<10; i++){
            std::cout << "byte number " << i << " " << int(buffer[i]) << "\n";
        }
        */

        /*

        start = high_resolution_clock::now();

        // Deserialize the frame
        frame = cv::imdecode(buffer, cv::IMREAD_COLOR);


        std::cout << "frame empty : " << frame.empty() << std::endl;

        if (frame.empty())
            return 1;

        */

        // Update the frame in the GUI
        updateFrame(received_image, frameLabel);
    });
    timer.start(2); // Update every 33 milliseconds (approximately 30 frames per second)

    // Show the main window
    window.show();

    return app.exec();
}


/*
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "opencvimageprovider.h"
#include "videostreamer.h"
#include <QQmlContext>
#include <paintitem.h>

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;

    qRegisterMetaType<cv::Mat>("cv::Mat");
    qmlRegisterType<PaintItem>("Painter", 1, 0, "PaintItem");

    VideoStreamer videoStreamer;

    OpencvImageProvider *liveImageProvider(new OpencvImageProvider);

    engine.rootContext()->setContextProperty("VideoStreamer",&videoStreamer);
    engine.rootContext()->setContextProperty("liveImageProvider",liveImageProvider);

    engine.addImageProvider("live",liveImageProvider);

    const QUrl url(QStringLiteral("qrc:/main.qml"));

    engine.load(url);

    QObject::connect(&videoStreamer,&VideoStreamer::newImage,liveImageProvider,&OpencvImageProvider::updateImage);

    return app.exec();
}
*/
