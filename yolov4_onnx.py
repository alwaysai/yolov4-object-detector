import edgeiq
from edgeiq import ObjectDetectionPostProcessParams, ObjectDetectionPreProcessParams
from typing import List
import numpy as np
import cv2


def yolov4_onnx_pre_process(params: ObjectDetectionPreProcessParams) -> np.ndarray:
    """Preprocessing on the CPU"""
    input_img = params.image
    print(f"[INFO] input_img.shape {input_img.shape}")
    # Model input
    resized = cv2.resize(input_img, params.size, interpolation=cv2.INTER_LINEAR)
    input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = np.transpose(input_tensor, (2, 0, 1)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor /= 255.0
    print("Shape of the network input: ", input_tensor.shape)
    return input_tensor


def yolov4_onnx_pre_process_trt(params: ObjectDetectionPreProcessParams) -> np.ndarray:
    input_img = params.image
    print(f"[INFO] input_img.shape {input_img.shape}")
    resized = cv2.resize(input_img, params.size, interpolation=cv2.INTER_LINEAR)
    input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # input_tensor = input_tensor / 255.0
    # input_tensor = input_img.transpose(2, 0, 1).ravel()
    input_tensor = np.transpose(input_tensor, (2, 0, 1)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor /= 255.0
    input_tensor = np.ascontiguousarray(input_tensor).ravel()
    print("Shape of the network input: ", input_tensor.shape)
    return input_tensor


def nms_cpu(boxes, confs, nms_thresh=0.5):
    """
    CPU-based Non-Maximum Suppression, Vectorized NMS Implementation
    using standard IoU as the overlap metric.  Data Structures: Assumes
    boxes input is a NumPy array of shape (num_boxes, 4), where the
    columns represent the coordinates (x1, y1, x2, y2).
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    # sort the indices based on the descending values in the array
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]  # i is the index of the highest remaining value in confs.
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Vectorized IoU calculation
        ious = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ious <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def yolov4_onnx_post_process(params: ObjectDetectionPostProcessParams):
    boxes: List[edgeiq.BoundingBox] = []
    confidences: List[float] = []
    indexes: List[int] = []

    outputs = params.results

    input_image: np.ndarray = params.image
    CONFIDENCE_THRESHOLD: float = params.confidence_level
    NMS_THRESHOLD: float = params.overlap_threshold
    image_height, image_width = input_image.shape[:2]
    # output format [batch, num, 1, 4]
    box_array = outputs[0]
    # output format [batch, num, num_classes]
    confs = outputs[1]
    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()
    box_array = box_array[:, :, 0]
    print(box_array)
    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
    # Vectorized filtering
    filtered_boxes = box_array[max_conf > CONFIDENCE_THRESHOLD]
    print(f"[INFO] initial filtered_boxes = {filtered_boxes}")
    # IMPORTANT: Reverse normalization from preprocessing
    filtered_boxes[:, [0, 2]] *= params.image.shape[1]
    filtered_boxes[:, [1, 3]] *= params.image.shape[0]
    filtered_scores = max_conf[max_conf > CONFIDENCE_THRESHOLD]
    filtered_classes = max_id[max_conf > CONFIDENCE_THRESHOLD]
    nms_keep = nms_cpu(filtered_boxes, filtered_scores, NMS_THRESHOLD)
    filtered_boxes = filtered_boxes[nms_keep]
    filtered_scores = filtered_scores[nms_keep]
    filtered_classes = filtered_classes[nms_keep]
    filtered_boxes = filtered_boxes.astype(int)
    print(f"[INFO] filtered_boxes = {filtered_boxes}")
    print(f"[INFO] filtered_scores = {filtered_scores}")
    print(f"[INFO] filtered_classes = {filtered_classes}")
    boxes = [edgeiq.BoundingBox(x[0], x[1], x[2], x[3]) for x in filtered_boxes]
    confidences = filtered_scores.tolist()
    indexes = filtered_classes.tolist()
    return boxes, confidences, indexes
