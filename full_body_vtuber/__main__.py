from mediapipe.python.solutions import pose
import cv2
from PIL import ImageDraw, Image
from pyvirtualcam import Camera
from argparse import ArgumentParser
import numpy
import math

parser = ArgumentParser()
parser.add_argument("-d", "--device", help="what camera device to use")
parser.add_argument("-b", "--camera-backend", help="what virtual camera backend to use")
parser.add_argument("-o", "--overlay", help="if enabled, displays the character over the camera image instead of drawing on a new canvas", action="store_true")
parser.add_argument("-dl", "--debug-landmarks", help="draws some landmarks that are useful for debugging", action="store_true")
parser.add_argument("-df", "--disable-filling", help="disables filling of drawn shapes", action="store_true")
parser.add_argument("-fi" "--fake-image", help="instead of camera frames, process a fake image of a standing person", action="store_true", dest="fake_image")

args = parser.parse_args()

if args.disable_filling:
    FILL = None
else:
    FILL = "white"
OUTLINE = "black"

video_capture = cv2.VideoCapture(0)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

LINE_WIDTH = 20

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def to_coords(landmark):
    return (landmark.x * width, landmark.y * height)

def extend(x1, y1, x2, y2, factor):
    x3 = x1 + (x2 - x1) * factor
    y3 = y1 + (y2 - y1) * factor
    x4 = x2 - (x2 - x1) * factor
    y4 = y2 - (y2 - y1) * factor
    return ((x3, y3), (x4, y4))

if args.fake_image:
    fake_image = numpy.asarray(Image.open("fake_image.jpeg").resize((width, height)))

with Camera(width=width, height=height, fps=int(video_capture.get(cv2.CAP_PROP_FPS)), device=args.device, backend=args.camera_backend) as camera:
    with pose.Pose() as pose_recognizer:
        while video_capture.isOpened():
            is_success, frame = video_capture.read()
            if args.fake_image:
                frame = fake_image
            if not is_success:
                break
            frame.flags.writeable = False # Makes it faster, from what I've heard
            output = pose_recognizer.process(frame)
            if output.pose_landmarks is not None:
                landmarks = output.pose_landmarks.landmark

                def landmark(index):
                    return to_coords(landmarks[index])

                if args.overlay:
                    if not args.fake_image:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    image = Image.new(mode="RGB", size=(width, height), color="white")
                draw = ImageDraw.ImageDraw(image)
                if args.debug_landmarks:
                    def debug_landmark(landmark_index):
                        x, y = landmark(landmark_index)
                        LANDMARK_RADIUS = 5
                        draw.ellipse(
                            (x - LANDMARK_RADIUS, y - LANDMARK_RADIUS, x + LANDMARK_RADIUS, y + LANDMARK_RADIUS),
                            fill="red"
                        )
                    debug_landmark(pose.PoseLandmark.LEFT_SHOULDER)
                    debug_landmark(pose.PoseLandmark.RIGHT_SHOULDER)
                    debug_landmark(pose.PoseLandmark.NOSE)
                    debug_landmark(pose.PoseLandmark.RIGHT_EAR)
                    debug_landmark(pose.PoseLandmark.LEFT_EAR)
                    debug_landmark(pose.PoseLandmark.LEFT_HIP)
                    debug_landmark(pose.PoseLandmark.RIGHT_HIP)
                nose = landmark(pose.PoseLandmark.NOSE)
                between_shoulders = midpoint(
                    *landmark(pose.PoseLandmark.RIGHT_SHOULDER),
                    *landmark(pose.PoseLandmark.LEFT_SHOULDER),
                )
                head_width = distance(*between_shoulders, *nose)
                head_radius = head_width / 2
                left_hip = landmark(pose.PoseLandmark.LEFT_HIP)
                right_hip = landmark(pose.PoseLandmark.RIGHT_HIP)
                left_hip, right_hip = extend(*left_hip, *right_hip, 1.5)
                hip_distance = distance(*left_hip, *right_hip)
                left_wrist = landmark(pose.PoseLandmark.LEFT_WRIST)
                #hand_distance = distance(*left_wrist, *to_coords(landmarks[pose.PoseLandmark.]))
                right_wrist = landmark(pose.PoseLandmark.RIGHT_WRIST)
                draw.polygon( # TORSO
                    (left_hip, right_hip, nose),
                    outline=OUTLINE, fill=FILL, width=LINE_WIDTH
                )
                draw.ellipse( # HEAD
                    (
                        nose[0] - head_radius,
                        nose[1] - head_radius,
                        nose[0] + head_radius,
                        nose[1] + head_radius,
                    ), outline=OUTLINE, fill=FILL, width=LINE_WIDTH
                )
                camera.send(numpy.asarray(image))
