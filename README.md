# Yolov4 ONNX Object Detector
YOLOv4 ONNX inferencing reference application allows developers to switch seamlessly between CPU and GPU. The script uses YOLOv4 ONNX Model from alwaysAI model catalog.
Application is designed to work with IP camera.  Simply add camera ip address as part of command line
arguments when starting the application:
```
aai app start -- --camera-url rtsp://acme:user@100.70.31.147/live
```
The script uses the on ONNX runtime for CPU inferencing engine (engine=edgeiq.Engine.ONNX_RT) or TENSORRT runtime (engine=edgeiq.Engine.TENSOR_RT) for NVIDIA GPU inferencing.  The YOLOv4 architecture is computationally heavy so if your application requires realtime performance use TENSORRT engine on a NVIDIA device.

## Requirements
* [alwaysAI account](https://alwaysai.co/auth?register=true)
* [alwaysAI Development Tools](https://alwaysai.co/docs/get_started/development_computer_setup.html)

## Usage
Once the alwaysAI tools are installed on your development machine (or edge device if developing directly on it) you can install and run the app with the following CLI commands:

To perform initial configuration of the app:
```
aai app configure
```

To prepare the runtime environment and install app dependencies:
```
aai app install
```

To start the app:
```
aai app start
```

To change the computer vision model, the engine and accelerator, and add additional dependencies read [this guide](https://alwaysai.co/docs/application_development/configuration_and_packaging.html).

## Support
* [Documentation](https://alwaysai.co/docs/)
* [Community Discord](https://discord.gg/alwaysai)
* Email: support@alwaysai.co
