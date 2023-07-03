import argparse

import requests
import json

import numpy as np
import cv2 as cv

from dasiamrpn import DaSiamRPN

# Check OpenCV version
assert cv.__version__ >= "4.7.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]
xPrev = 0
yPrev = 0

parser = argparse.ArgumentParser(
    description="Distractor-aware Siamese Networks for Visual Object Tracking (https://arxiv.org/abs/1808.06048)")
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input video. Omit for using default camera.')
parser.add_argument('--model_path', type=str, default='object_tracking_dasiamrpn_model_2021nov.onnx',
                    help='Usage: Set model path, defaults to object_tracking_dasiamrpn_model_2021nov.onnx.')
parser.add_argument('--kernel_cls1_path', type=str, default='object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx',
                    help='Usage: Set path to dasiamrpn_kernel_cls1.onnx.')
parser.add_argument('--kernel_r1_path', type=str, default='object_tracking_dasiamrpn_kernel_r1_2021nov.onnx',
                    help='Usage: Set path to dasiamrpn_kernel_r1.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save a file with results. Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def visualize(image, bbox, score, isLocated, fps=None, box_color=(0, 255, 0),text_color=(0, 255, 0), fontScale = 1, fontSize = 1):
    global xPrev
    global yPrev
    output = image.copy()
    h, w, _ = output.shape

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 30), cv.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)

    if isLocated and score >= 0.75:
        # bbox: Tuple of length 4
        x, y, w, h = bbox
        cv.rectangle(output, (x, y), (x+w, y+h), box_color, 2)
        cv.putText(output, '{:.2f}'.format(score), (0, 60), cv.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)
        r = requests.post('http://di205523.beget.tech/setAI.php', data = json.dumps({'x': (w)/2+x, 'y': (h)/2+y, 'data': 1}), headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (platform; rv:geckoversion) Gecko/geckotrail Firefox/firefoxversion'})
        xPrev = w/2+x
        yPrev = h/2+y
        # print(x)
        # print(y)
        # print()
        # print(x+w)
        # print(y)
        # print()
        # print(x+w)
        # print(y+h)
        # print()
        # print(x)
        # print(y+h)
        # print("--")
        
    else:
        text_size, baseline = cv.getTextSize('Animal lost', cv.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = int((w - text_size[0]) / 2)
        cv.putText(output, 'Animal lost', (text_x, 30), cv.FONT_HERSHEY_DUPLEX, fontScale, (0, 0, 255), fontSize)
        r = requests.post('http://di205523.beget.tech/setAI.php', data = json.dumps({'x': xPrev, 'y': yPrev, 'data': 0}), headers={
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'User-Agent': 'Mozilla/5.0 (platform; rv:geckoversion) Gecko/geckotrail Firefox/firefoxversion'})

    return output

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate DaSiamRPN
    model = DaSiamRPN(
        kernel_cls1_path=args.kernel_cls1_path,
        kernel_r1_path=args.kernel_r1_path,
        model_path=args.model_path,
        backend_id=backend_id,
        target_id=target_id)

    # Read from args.input
    _input = args.input
    if args.input is None:
        device_id = 0
        _input = device_id
    video = cv.VideoCapture(_input)

    # Select an object
    has_frame, first_frame = video.read()
    if not has_frame:
        print('No frames grabbed!')
        exit()
    first_frame_copy = first_frame.copy()
    cv.putText(first_frame_copy, "1. Drag a bounding box to track.", (0, 15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv.putText(first_frame_copy, "2. Press ENTER to confirm", (0, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    roi = cv.selectROI('DaSiamRPN Demo', first_frame_copy)
    print("Selected ROI: {}".format(roi))

    # Init tracker with ROI
    model.init(first_frame, roi)

    # Track frame by frame
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        has_frame, frame = video.read()
        if not has_frame:
            print('End of video')
            break
        # Inference
        tm.start()
        isLocated, bbox, score = model.infer(frame)
        tm.stop()
        # Visualize
        frame = visualize(frame, bbox, score, isLocated, fps=tm.getFPS())
        cv.imshow('DaSiamRPN Demo', frame)
        tm.reset()
