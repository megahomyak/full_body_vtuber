from mediapipe.python.solutions import pose
import cv2
from PIL import ImageDraw, Image
from pyvirtualcam import Camera
from argparse import ArgumentParser
import numpy

parser = ArgumentParser()
parser.add_argument("-d", "--device", help="what camera device to use")
parser.add_argument("-b", "--camera-backend", help="what virtual camera backend to use")

args = parser.parse_args()

video_capture = cv2.VideoCapture(0)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

LINE_WIDTH = 20

def midpoint(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y + landmark2.y) ** 2) ** 0.5

with Camera(width=width, height=height, fps=60, device=args.device, backend=args.camera_backend) as camera:
    with pose.Pose() as pose_recognizer:
        while video_capture.isOpened():
            is_success, frame = video_capture.read()
            if not is_success:
                break
            frame.flags.writeable = False # Makes it faster, from what I've heard
            output = pose_recognizer.process(frame)
            if output.pose_landmarks is not None:
                landmarks = output.pose_landmarks.landmark
                image = Image.new(mode="RGB", size=(width, height), color="white")
                draw = ImageDraw.ImageDraw(image)
                head = landmarks[pose.PoseLandmark.NOSE]
                print(head.z)
                head_radius = (head.z + 2) * 50
                draw.ellipse(
                    (
                        width * head.x - head_radius,
                        height * head.y - head_radius,
                        width * head.x + head_radius,
                        height * head.y + head_radius,
                    ), outline="black", width=LINE_WIDTH
                )
                camera.send(numpy.asarray(image))
