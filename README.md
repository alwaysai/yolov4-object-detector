# Yolov4 ONNX Object Detector
This repo contains a YOLOv4 ONNX reference application script that switches seamlessly between CPU and GPU inferencing. The script uses a YOLOv4 ONNX model created from the coco dataset and can be found in alwaysAI's model catalog. Script works with IP cameras.  To run simply add your ip camera's address as part of command line arguments when starting the application:
```
aai app start -- --camera-url rtsp://acme:user@100.70.31.147/live
```
This script uses either alwaysAI's ONNX runtime for CPU inferencing  ```engine=edgeiq.Engine.ONNX_RT``` or TENSORRT runtime ```engine=edgeiq.Engine.TENSOR_RT``` for GPU inferencing.  Adjust the script in ```app.py``` to met your inferencing needs.  The YOLOv4 architecture is computationally heavy so if your application requires realtime performance use the TENSORRT engine.

### alwaysAI inference engine information

## Requirements
* [alwaysAI account](https://alwaysai.co/auth?register=true)
* [alwaysAI Development Tools](https://alwaysai.co/docs/get_started/development_computer_setup.html)

## Usage
Once the alwaysAI tools are installed on your development machine (or edge device if developing directly on it) you can install and run the app with the following CLI commands:

To perform initial configuration of the app:
```
aai app configure
```
Add model to the app:
```
aai app models add alwaysai/yolov4-onnx
```
To prepare the runtime environment and install app dependencies:
```
aai app install
```

To start the app (customize to your camera url):
```
aai app start -- --camera-url rtsp://acme:user@100.70.31.147/live
```
To change the computer vision model, the engine and accelerator, and add additional dependencies read [this guide](https://alwaysai.co/docs/application_development/configuration_and_packaging.html).

### NIVIDA x86 GTX 1650 GPU Usage Information
alwaysAI also supports inferencing on GeForce 1650 GPU's for x86 devices.  To perform initial configuration of the app:
```
aai app configure --hardware x86-trt-23.02
```
Set ```ALWAYSAI_HW to x86-trt-23.02 in your Dockerfile```
Then follow the Usage instructions starting with the ```aai add models``` command.

## Support
* [Documentation](https://alwaysai.co/docs/)
* [Community Discord](https://discord.gg/alwaysai)
* Email: support@alwaysai.co
