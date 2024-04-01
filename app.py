
"""
YOLOv4 ONNX inferencing reference application.  Uses YOLOv4 ONNX Model from alwaysAI model catalog.
Application is designed to work with IP camera.  Simply add camera ip address as part of command line
arguments when starting the application:  aai app start -- --camera-url rtsp://acme:user@100.70.31.147/live
The application can run on ONNX (engine=edgeiq.Engine.ONNX_RT) or TENSORRT (engine=edgeiq.Engine.TENSOR_RT) runtime
engine.
"""
import time
import os
import argparse
os.environ["ALWAYSAI_DBG_DISABLE_MODEL_VALIDATION"] = "1"
import edgeiq
from yolov4_onnx import yolov4_onnx_pre_process_trt, yolov4_onnx_post_process


def main(camera_url):
    obj_detect = edgeiq.ObjectDetection("alwaysai/yolov4-onnx", pre_process=yolov4_onnx_pre_process_trt,
                                        post_process=yolov4_onnx_post_process)
    # Change engine to engine=edgeiq.Engine.ONNX_RT for CPU inferencing
    obj_detect.load(engine=edgeiq.Engine.TENSOR_RT)

    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.IPVideoStream(camera_url) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level=.5)
                frame = edgeiq.markup_image(frame, results.predictions, colors=obj_detect.colors)

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append("Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                for prediction in results.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera IP address')
    parser.add_argument("--camera-url", default="", type=str, required=True)
    args = parser.parse_args()
    main(camera_url=args.camera_url)
