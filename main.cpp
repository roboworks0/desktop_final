#include <QApplication>
#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>
#include <QPushButton>
#include <QFile>
#include <QTextStream>
#include <QDebug>
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
    // Load the pre-trained YOLO model
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

    // Convert the frame to QImage
    QImage image(frame.data, frame.cols, frame.rows, QImage::Format_RGB888);
    image = image.rgbSwapped();

    // Set the QImage as the pixmap of the frame label widget
    frameLabel->setPixmap(QPixmap::fromImage(image));
    frameLabel->setScaledContents(true);
}

int main(int argc, char *argv[])
{
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

    // Create random buttons
    for (int i = 0; i < 5; i++) {
        QPushButton* button = new QPushButton(QString("Button %1").arg(i + 1), centralWidget);
        layout->addWidget(button);
    }

    // Set the layout to the central widget
    centralWidget->setLayout(layout);

    // Initialize the server communication and receive the frame
    // Replace this part with your server communication code from server.cpp




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

    // Accept incoming connection
    if ((new_socket = accept(server_socket, (struct sockaddr *) &address, (socklen_t *) &addrlen)) < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }

    std::vector<uchar> buffer;
    int message_size;

    std::cout << "test..." << "\n";
    int counter = 0;


    cv::Mat frame;



    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, [&]() {
        //cap >> frame;

        std::cout << "debug1\n";
        if (read(new_socket, &message_size, sizeof(message_size)) == 0) {
            perror("Client disconnected");
            return 1;
        }
        std::cout << "debug2\n";

        std::cout << "message size : " << message_size << "\n";

        // Resize the buffer to fit the received frame
        buffer.resize(message_size);

        auto start = high_resolution_clock::now();
        ssize_t bytes_read;
        ssize_t total_bytes_read = 0;
        while (total_bytes_read < message_size)
        {
            std::cout << "total_bytes_read : " << total_bytes_read << "\n";
            ssize_t bytes_read = read(new_socket, buffer.data() + total_bytes_read, message_size - total_bytes_read);
            std::cout << "bytes_read : " << bytes_read << "\n";
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
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Time taken by function: "
            << duration.count() * 0.001 << " miliseconds" << endl;


        std::cout << "bytes_read : " << bytes_read << "\n";
        for(int i=0; i<10; i++){
            std::cout << "byte number " << i << " " << int(buffer[i]) << "\n";
        }

        start = high_resolution_clock::now();

        // Deserialize the frame
        frame = cv::imdecode(buffer, cv::IMREAD_COLOR);


        if (frame.empty())
            return 1;

        // Update the frame in the GUI
        updateFrame(frame, frameLabel);
    });
    timer.start(33); // Update every 33 milliseconds (approximately 30 frames per second)

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
