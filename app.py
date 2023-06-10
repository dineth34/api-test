from flask import Flask,request
import urllib.request
import os
from firebase_admin import credentials, initialize_app, storage
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import csv
from sklearn.metrics import accuracy_score,silhouette_score 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import ast
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import random
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import pandas as pd
from sklearn.model_selection import train_test_split



app = Flask(__name__)

stores = [{"name": "My Store", "items": [{"name": "my item", "price": 15.99}]}]

def get_file_details():
    current_location = os.getcwd()  # Get the current working directory
    
    file_details = []  # List to store file details
    
    for root, dirs, files in os.walk(current_location):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            file_details.append({
                'file_name': file,
                'file_path': file_path,
                'file_size': file_size
            })
    
    return file_details

def init_firestore():
    cred = credentials.Certificate("t-13859-firebase-adminsdk-eu692-ac3180a413.json")
    initialize_app(cred, {'storageBucket': 't-13859.appspot.com'})

def download_file_firestore(bucket_name, file_name, destination_path):

    # Get the bucket reference
    bucket = storage.bucket(bucket_name)

    # Specify the file path in the bucket
    blob = bucket.blob(file_name)

    # Download the file to the specified destination path
    blob.download_to_filename(destination_path)


def download_files():
    urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg', 'yolov4.cfg')
    urllib.request.urlretrieve('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights', 'yolov4.weights')
    urllib.request.urlretrieve('https://drive.google.com/uc?id=1UTuoWFmOjyn-SFOhhF0rWtl1gTU9Yk_3&export=download', 't-13859-firebase-adminsdk-eu692-ac3180a413.json')
    # urllib.request.urlretrieve('https://drive.google.com/uc?id=16YxH5z1mOIzmL1s6nOdbLxJshpn8_Hbg&export=download','padded_matrix_file.csv')
    init_firestore()
    download_file_firestore('t-13859.appspot.com','TE-64_alpha_0.9.pt','TE-64_alpha_0.9.pt')
    download_file_firestore('t-13859.appspot.com','trained_model.pt','trained_model.pt')
    download_file_firestore('t-13859.appspot.com','best_z_proto.pt','best_z_proto.pt')

download_files()

def convert_30_fps(file_path):
    fps_converted_videos_path = "30fps"
    
    if not os.path.exists(fps_converted_videos_path):
        os.makedirs(fps_converted_videos_path)
    
    # Open input video
    cap = cv2.VideoCapture(file_path)

    
    # Get input video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define output video properties
    out_fps = 30
    out_frame_width = frame_width
    out_frame_height = frame_height

    # Create output video writer
    file_name = file_path.split('/')[-1]
    out_filepath_30_fps = os.path.join(fps_converted_videos_path, file_name)
    out = cv2.VideoWriter(out_filepath_30_fps, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (out_frame_width, out_frame_height))

    # Read and write each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release input and output video resources
    cap.release()
    out.release()
    return out_filepath_30_fps

def draw_bounding_box(video_path):
    # create the output directory if it doesn't exist
    output_dir = "videos_with_bouding_boxes"
    os.makedirs(output_dir, exist_ok=True)

    # initialize YOLOv4 object detection model
    model = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # get the names of the output layers
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    # open the video file
    cap = cv2.VideoCapture(video_path)

    # get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create the output video writer
    output_path_videos_with_bounding_box = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path_videos_with_bounding_box, fourcc, fps, (width, height))

    # loop through all the frames
    while True:
        # read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # create a blob from the frame and pass it through the YOLOv4 model
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
        model.setInput(blob)
        outputs = model.forward(output_layers)

        # get the bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0: # check for red color
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # apply non-maximum suppression to suppress weak, overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # draw the bounding boxes and class labels on the frame
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                color = (0, 0, 255) # set color to red
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # write the frame to the output video
        out.write(frame)

    # release the video capture and output video writer
    cap.release()
    out.release()

    return output_path_videos_with_bounding_box

def get_bounding_boxes(video_path):
    total_coordinates=[]
    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video metadata
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create list to hold bounding boxes for each frame
    boxes_per_frame = []

    # Loop through frames
    for i in range(num_frames):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding to extract red regions
        lower_red = np.array([0,0,200])
        upper_red = np.array([50,50,255])
        mask = cv2.inRange(frame, lower_red, upper_red)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find bounding box for each contour
        bboxes = []
        for contour in contours:
            frame_coordinates=[]
            x, y, w, h = cv2.boundingRect(contour)
            if x >= 0 and y >= 0:
                bboxes.append([x, y, w, h])
            topLeft=[x,y]
            topRight=[x+w,y]
            bottomLeft=[x,y+h]
            bottomRight=[x+w,y+h]
            frame_coordinates.append(topLeft)
            frame_coordinates.append(topRight)
            frame_coordinates.append(bottomLeft)
            frame_coordinates.append(bottomRight)
            total_coordinates.append(frame_coordinates)
        # Append list of bounding boxes to main list
        boxes_per_frame.append(bboxes)

    # Release video capture object
    cap.release()

    return total_coordinates

def find_bounding_box(rectangles):
    x_coords = [p[0] for r in rectangles for p in r]
    y_coords = [p[1] for r in rectangles for p in r]
  
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    tl = (x_min, y_min)
    tr = (x_max, y_min)
    bl = (x_min, y_max)
    br = (x_max, y_max)
    return [tl, tr, bl, br]

def crop_video(video_path, coords):
    
    # create the output directory if it doesn't exist
    output_dir = "cropped_videos"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Define the codec and create VideoWriter object
    cropped_video_path = os.path.join(output_dir, os.path.basename(video_path))
    out = cv2.VideoWriter(cropped_video_path,fourcc, fps, (coords[2] - coords[0], coords[3] - coords[1]))

    # Iterate over video frames and crop each frame
    while success:
        # Crop image to only include bounding box region
        cropped_image = image[coords[1]:coords[3], coords[0]:coords[2]]

        # Write the cropped image to the output video file
        out.write(cropped_image)

        # Read the next frame
        success, image = cap.read()

    # Release video and output file
    cap.release()
    out.release()

    return cropped_video_path

def normalize_coordinates(coordinates, frame_width, frame_height):
    normalized_coordinates = []
    for x, y in coordinates:
        normalized_x = x / float(frame_width)
        normalized_y = y / float(frame_height)
        normalized_coordinates.append((normalized_x, normalized_y))
    return normalized_coordinates

def extract_skeleton_locations(video_file):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_file)
    csv_file = open('skeleton_locations.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['file_name','nose', 'left_eye_inner','left_eye', 
                         'left_eye_outer', 'right_eye_inner', 
                         'right_eye', 'right_eye_outer' ,'left_ear',
                         'right_ear', 'mouth_left', 'mouth_right',
                         'left_elbow', 'right_elbow','right_shoulder', 'left_shoulder', 'left_pinky_tip', 'right_pinky_tip',
                        'left_wrist','left_thumb_cmc','left_thumb_mcp','left_thumb_ip',
    'left_thumb_tip','left_index_finger_mcp','left_index_finger_pip','left_index_finger_dip',
    'left_index_finger_tip','left_middle_finger_mcp','left_middle_finger_pip',
    'left_middle_finger_dip','left_middle_finger_tip','left_ring_finger_mcp',
    'left_ring_finger_pip','left_ring_finger_dip','left_ring_finger_tip',
    'left_pinky_mcp','left_pinky_pip','left_pinky_dip',
    'right_wrist','right_thumb_cmc','right_thumb_mcp',
    'right_thumb_ip','right_thumb_tip','right_index_finger_mcp',
    'right_index_finger_pip','right_index_finger_dip','right_index_finger_tip',
    'right_middle_finger_mcp','right_middle_finger_pip',
    'right_middle_finger_dip','right_middle_finger_tip',
    'right_ring_finger_mcp','right_ring_finger_pip','right_ring_finger_dip',
    'right_ring_finger_tip','right_pinky_mcp','right_pinky_pip',
    'right_pinky_dip'])
    print("csv created")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame)
        results_hands = hands.process(frame)
        annotated_image = frame.copy()
#         if results_pose.pose_landmarks:
#             mp_drawing.draw_landmarks(annotated_image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         if results_hands.multi_hand_landmarks:
#             for hand_landmarks in results_hands.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        file_name = video_file.split('/')[-1]
        # Body
        right_shoulder_locations = []
        left_shoulder_locations = []
        nose_locations = []    
        left_eye_inner_locations = []
        left_eye_locations = []
        left_eye_outer_locations = []
        right_eye_inner_locations = []
        right_eye_locations = []
        right_eye_outer_locations = []
        left_ear_locations = []
        right_ear_locations = []
        mouth_left_locations = []
        mouth_right_locations = []
        left_elbow_locations = []
        right_elbow_locations = []
        #hand
        left_pinky_tip_locations = []
        right_pinky_tip_locations = []
        left_wrist_locations = []
        left_thumb_cmc_locations = []
        left_thumb_mcp_locations = []
        left_thumb_ip_locations = []
        left_thumb_tip_locations = []
        left_index_finger_mcp_locations = []
        left_index_finger_pip_locations = []
        left_index_finger_dip_locations = []
        left_index_finger_tip_locations = []
        left_middle_finger_mcp_locations = []
        left_middle_finger_pip_locations = []
        left_middle_finger_dip_locations = []
        left_middle_finger_tip_locations = []
        left_ring_finger_mcp_locations = []
        left_ring_finger_pip_locations = []
        left_ring_finger_dip_locations = []
        left_ring_finger_tip_locations = []
        left_pinky_mcp_locations = []
        left_pinky_pip_locations = []
        left_pinky_dip_locations = []
        
        right_wrist_locations = []
        right_thumb_cmc_locations = []
        right_thumb_mcp_locations = []
        right_thumb_ip_locations = []
        right_thumb_tip_locations = []
        right_index_finger_mcp_locations = []
        right_index_finger_pip_locations = []
        right_index_finger_dip_locations = []
        right_index_finger_tip_locations = []
        right_middle_finger_mcp_locations = []
        right_middle_finger_pip_locations = []
        right_middle_finger_dip_locations = []
        right_middle_finger_tip_locations = []
        right_ring_finger_mcp_locations = []
        right_ring_finger_pip_locations = []
        right_ring_finger_dip_locations = []
        right_ring_finger_tip_locations = []
        right_pinky_mcp_locations = []
        right_pinky_pip_locations = []
        right_pinky_dip_locations = []
        #hand both
        wrist_locations = []
        thumb_cmc_locations = []
        thumb_mcp_locations = []
        thumb_ip_locations = []
        thumb_tip_locations = []
        index_finger_mcp_locations = []
        index_finger_pip_locations = []
        index_finger_dip_locations = []
        index_finger_tip_locations = []
        middle_finger_mcp_locations = []
        middle_finger_pip_locations = []
        middle_finger_dip_locations = []
        middle_finger_tip_locations = []
        ring_finger_mcp_locations = []
        ring_finger_pip_locations = []
        ring_finger_dip_locations = []
        ring_finger_tip_locations = []
        pinky_tip_locations = []
        pinky_mcp_locations = []
        pinky_pip_locations = []
        pinky_dip_locations = []
        if results_pose.pose_landmarks:
            for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if landmark.visibility < 0.05:
                    landmark_px = (0, 0)
                else:
                    landmark_px = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                if not landmark_px:
                    landmark_px = (0, 0)
                # Normalize the landmark coordinates
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                normalized_landmark_px = normalize_coordinates([landmark_px], frame_width, frame_height)[0]
                landmark_px = normalized_landmark_px
                if i == mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                    right_shoulder_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.LEFT_SHOULDER.value:
                    left_shoulder_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.NOSE.value:
                    nose_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.LEFT_EYE_INNER.value:
                    left_eye_inner_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.LEFT_EYE.value:
                    left_eye_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.LEFT_EYE_OUTER.value:
                    left_eye_outer_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.RIGHT_EYE_INNER.value:
                    right_eye_inner_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.RIGHT_EYE.value:
                    right_eye_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value:
                    right_eye_outer_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.LEFT_EAR.value:
                    left_ear_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.RIGHT_EAR.value:
                    right_ear_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.MOUTH_LEFT.value:
                    mouth_left_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.MOUTH_RIGHT.value:
                    mouth_right_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.LEFT_ELBOW.value:
                    left_elbow_locations.append(landmark_px)
                elif i == mp_pose.PoseLandmark.RIGHT_ELBOW.value:
                    right_elbow_locations.append(landmark_px)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    landmark_px = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]
                    normalized_landmark_px = normalize_coordinates([landmark_px], frame_width, frame_height)[0]
                    landmark_px = normalized_landmark_px
                    if i == mp_hands.HandLandmark.PINKY_TIP.value:
                        pinky_tip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.WRIST.value:
                        wrist_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.THUMB_CMC.value:
                        thumb_cmc_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.THUMB_MCP.value:
                        thumb_mcp_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.THUMB_IP.value:
                        thumb_ip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.THUMB_TIP.value:
                        thumb_tip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.INDEX_FINGER_MCP.value:
                        index_finger_mcp_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.INDEX_FINGER_PIP.value:
                        index_finger_pip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.INDEX_FINGER_DIP.value:
                        index_finger_dip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.INDEX_FINGER_TIP.value:
                        index_finger_tip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value:
                        middle_finger_mcp_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value:
                        middle_finger_pip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.MIDDLE_FINGER_DIP.value:
                        middle_finger_dip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value:
                        middle_finger_tip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.RING_FINGER_MCP.value:
                        ring_finger_mcp_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.RING_FINGER_PIP.value:
                        ring_finger_pip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.RING_FINGER_DIP.value:
                        ring_finger_dip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.RING_FINGER_TIP.value:
                        ring_finger_tip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.PINKY_MCP.value:
                        pinky_mcp_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.PINKY_PIP.value:
                        pinky_pip_locations.append(landmark_px)
                    elif i == mp_hands.HandLandmark.PINKY_DIP.value:
                        pinky_dip_locations.append(landmark_px)
        if len(pinky_tip_locations) > 0:
            right_pinky_tip_locations = [pinky_tip_locations[0]]
        else:
            right_pinky_tip_locations = [(0,0)]
        if len(pinky_tip_locations) > 1:
            left_pinky_tip_locations = [pinky_tip_locations[1]]
        else:
            left_pinky_tip_locations = [(0,0)]
            #################################
        if len(wrist_locations) > 0:
            right_wrist_locations = [wrist_locations[0]]
        else:
            right_wrist_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_wrist_locations = [wrist_locations[1]]
        else:
            left_wrist_locations = [(0,0)]
            ################################
        if len(wrist_locations) > 0:
            right_thumb_cmc_locations = [thumb_cmc_locations[0]]
        else:
            right_thumb_cmc_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_thumb_cmc_locations = [thumb_cmc_locations[1]]
        else:
            left_thumb_cmc_locations = [(0,0)]
           ################################
        if len(wrist_locations) > 0:
            right_thumb_mcp_locations = [thumb_mcp_locations[0]]
        else:
            right_thumb_mcp_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_thumb_mcp_locations = [thumb_mcp_locations[1]]
        else:
            left_thumb_mcp_locations = [(0,0)]
           ################################
        if len(wrist_locations) > 0:
            right_thumb_ip_locations = [thumb_ip_locations[0]]
        else:
            right_thumb_ip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_thumb_ip_locations = [thumb_ip_locations[1]]
        else:
            left_thumb_ip_locations = [(0,0)]
           ################################
        if len(wrist_locations) > 0:
            right_thumb_tip_locations = [thumb_tip_locations[0]]
        else:
            right_thumb_tip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_thumb_tip_locations = [thumb_tip_locations[1]]
        else:
            left_thumb_tip_locations = [(0,0)]
          ################################
        if len(wrist_locations) > 0:
            right_index_finger_mcp_locations = [index_finger_mcp_locations[0]]
        else:
            right_index_finger_mcp_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_index_finger_mcp_locations = [index_finger_mcp_locations[1]]
        else:
            left_index_finger_mcp_locations = [(0,0)]
         ################################
        if len(wrist_locations) > 0:
            right_index_finger_pip_locations = [index_finger_pip_locations[0]]
        else:
            right_index_finger_pip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_index_finger_pip_locations = [index_finger_pip_locations[1]]
        else:
            left_index_finger_pip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_index_finger_dip_locations = [index_finger_dip_locations[0]]
        else:
            right_index_finger_dip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_index_finger_dip_locations = [index_finger_dip_locations[1]]
        else:
            left_index_finger_dip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_index_finger_tip_locations = [index_finger_tip_locations[0]]
        else:
            right_index_finger_tip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_index_finger_tip_locations = [index_finger_tip_locations[1]]
        else:
            left_index_finger_tip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_middle_finger_mcp_locations = [middle_finger_mcp_locations[0]]
        else:
            right_middle_finger_mcp_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_middle_finger_mcp_locations = [middle_finger_mcp_locations[1]]
        else:
            left_middle_finger_mcp_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_middle_finger_pip_locations = [middle_finger_pip_locations[0]]
        else:
            right_middle_finger_pip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_middle_finger_pip_locations = [middle_finger_pip_locations[1]]
        else:
            left_middle_finger_pip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_middle_finger_dip_locations = [middle_finger_dip_locations[0]]
        else:
            right_middle_finger_dip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_middle_finger_dip_locations = [middle_finger_dip_locations[1]]
        else:
            left_middle_finger_dip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_middle_finger_tip_locations = [middle_finger_tip_locations[0]]
        else:
            right_middle_finger_tip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_middle_finger_tip_locations = [middle_finger_tip_locations[1]]
        else:
            left_middle_finger_tip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_ring_finger_mcp_locations = [ring_finger_mcp_locations[0]]
        else:
            right_ring_finger_mcp_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_ring_finger_mcp_locations = [ring_finger_mcp_locations[1]]
        else:
            left_ring_finger_mcp_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_ring_finger_pip_locations = [ring_finger_pip_locations[0]]
        else:
            right_ring_finger_pip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_ring_finger_pip_locations = [ring_finger_pip_locations[1]]
        else:
            left_ring_finger_pip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_ring_finger_dip_locations = [ring_finger_dip_locations[0]]
        else:
            right_ring_finger_dip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_ring_finger_dip_locations = [ring_finger_dip_locations[1]]
        else:
            left_ring_finger_dip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_ring_finger_tip_locations = [ring_finger_tip_locations[0]]
        else:
            right_ring_finger_tip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_ring_finger_tip_locations = [ring_finger_tip_locations[1]]
        else:
            left_ring_finger_tip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_pinky_mcp_locations = [pinky_mcp_locations[0]]
        else:
            right_pinky_mcp_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_pinky_mcp_locations = [pinky_mcp_locations[1]]
        else:
            left_pinky_mcp_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_pinky_pip_locations = [pinky_pip_locations[0]]
        else:
            right_pinky_pip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_pinky_pip_locations = [pinky_pip_locations[1]]
        else:
            left_pinky_pip_locations = [(0,0)]
        ################################
        if len(wrist_locations) > 0:
            right_pinky_dip_locations = [pinky_dip_locations[0]]
        else:
            right_pinky_dip_locations = [(0,0)]
        if len(wrist_locations) > 1:
            left_pinky_dip_locations = [pinky_dip_locations[1]]
        else:
            left_pinky_dip_locations = [(0,0)]
        nose_locations = nose_locations if nose_locations else [(0,0)]
        left_eye_inner_locations = left_eye_inner_locations if left_eye_inner_locations else [(0,0)]
        left_eye_locations = left_eye_locations if left_eye_locations else [(0,0)]
        left_eye_outer_locations = left_eye_outer_locations if left_eye_outer_locations else [(0,0)]
        right_eye_inner_locations = right_eye_inner_locations if right_eye_inner_locations else [(0,0)]
        right_eye_locations = right_eye_locations if right_eye_locations else [(0,0)]
        right_eye_outer_locations = right_eye_outer_locations if right_eye_outer_locations else [(0,0)]
        left_ear_locations = left_ear_locations if left_ear_locations else [(0,0)]
        right_ear_locations = right_ear_locations if right_ear_locations else [(0,0)]
        mouth_left_locations = mouth_left_locations if mouth_left_locations else [(0,0)]
        mouth_right_locations = mouth_right_locations if mouth_right_locations else [(0,0)]
        left_elbow_locations = left_elbow_locations if left_elbow_locations else [(0,0)]
        right_elbow_locations = right_elbow_locations if right_elbow_locations else [(0,0)]
        right_shoulder_locations = right_shoulder_locations if right_shoulder_locations else [(0,0)]
        left_shoulder_locations = left_shoulder_locations if left_shoulder_locations else [(0,0)]
        writer.writerow([file_name,nose_locations, left_eye_inner_locations,left_eye_locations, 
                         left_eye_outer_locations, right_eye_inner_locations, 
                         right_eye_locations, right_eye_outer_locations ,left_ear_locations,
                         right_ear_locations, mouth_left_locations, mouth_right_locations,
                         left_elbow_locations, right_elbow_locations ,right_shoulder_locations, left_shoulder_locations, left_pinky_tip_locations, right_pinky_tip_locations,
                        left_wrist_locations,left_thumb_cmc_locations,left_thumb_mcp_locations,left_thumb_ip_locations,
    left_thumb_tip_locations,left_index_finger_mcp_locations,left_index_finger_pip_locations,left_index_finger_dip_locations,
    left_index_finger_tip_locations,left_middle_finger_mcp_locations,left_middle_finger_pip_locations,
    left_middle_finger_dip_locations,left_middle_finger_tip_locations,left_ring_finger_mcp_locations,
    left_ring_finger_pip_locations,left_ring_finger_dip_locations,left_ring_finger_tip_locations,
    left_pinky_mcp_locations,left_pinky_pip_locations,left_pinky_dip_locations,
    right_wrist_locations,right_thumb_cmc_locations,right_thumb_mcp_locations,
    right_thumb_ip_locations,right_thumb_tip_locations,right_index_finger_mcp_locations,
    right_index_finger_pip_locations,right_index_finger_dip_locations,right_index_finger_tip_locations,
    right_middle_finger_mcp_locations,right_middle_finger_pip_locations,
    right_middle_finger_dip_locations,right_middle_finger_tip_locations,
    right_ring_finger_mcp_locations,right_ring_finger_pip_locations,right_ring_finger_dip_locations,
    right_ring_finger_tip_locations,right_pinky_mcp_locations,right_pinky_pip_locations,
    right_pinky_dip_locations])

def matrix_creation(input_path):
    output_path = 'output_XY_file.csv'

    # Open the input CSV file and the output CSV file
    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:

        # Create a CSV reader object and a CSV writer object
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        next(reader)

        # Iterate over each row in the input CSV file
        for row in reader:

            transformed_row = []

            file_name = row[0]

            for cell in row[1:]:

                # Get the value from the current cell
                cell_value = cell

                if cell_value == 'file_name':
                    continue

                numbers = cell_value.strip("[]()").split(",")

                result = tuple(float(x.strip("()[] ")) for x in numbers)

                # Extract the two values from the cell value
                value1, value2 = result

                transformed_row.append(value1)
                transformed_row.append(value2)

            transformed_row.insert(0, file_name)

            # Write the two values to a new row in the output CSV file
            writer.writerow(transformed_row)
            
    # Specify the input and output file paths
    input_path_2 = 'output_XY_file.csv'
    output_path_2 = 'output_SingleCell_file.csv'

    # Open the input CSV file and the output CSV file
    with open(input_path_2, 'r') as input_file, open(output_path_2, 'w', newline='') as output_file:

        # Create a CSV reader object and a CSV writer object
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        next(reader)

        # Iterate over each row in the input CSV file
        for row in reader:

            transformed_row = []

            file_name = row[0]

            for cell in row[1:]:

                # Get the value from the current cell
                cell_value = cell

                if cell_value == 'file_name':
                    continue

                transformed_row.append(float(cell_value))

            # Write the two values to a new row in the output CSV file
            writer.writerow([file_name, transformed_row])
            
    
    # Load the CSV file
    with open('output_SingleCell_file.csv', 'r') as f:
        reader = csv.reader(f)

        data = {}
        for row in reader:
            filename = row[0]
            skeleton = row[1]
            if filename in data:
                data[filename].append(skeleton)
            else:
                data[filename] = [skeleton]
    #     print(data)

    # Write the grouped data to a new CSV file
    with open('matrix_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for filename, skeletons in data.items():

            # number_str = filename[:3]  # extract the first three characters
            # number = int(number_str)  # convert to integer
            # label = number - 1  # subtract one

            skeletons.insert(0,filename)
            writer.writerow(skeletons)
            
    max_length_video = 243
    
    # Specify the input and output file paths
    input_path_3 = 'matrix_file.csv'
    output_path_3 = 'padded_matrix_file.csv'

    # Open the input CSV file and the output CSV file
    with open(input_path_3, 'r') as input_file, open(output_path_3, 'w', newline='') as output_file:

        # Create a CSV reader object and a CSV writer object
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Iterate over each row in the input CSV file
        for row in reader:

            if( len(row)<max_length_video ):

                pad_item = len(eval(row[1]))*[0]

                add_list = (max_length_video-len(row))*[pad_item]

                row = row + add_list

            # Write the two values to a new row in the output CSV file

            row = row[1:] + [row[0]]

            writer.writerow(row)

    print("Final Matrix Creation Successful")

class TransformerEncoder(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=16, num_layers=1,n_classes=64):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.positional_encoding = self.generate_positional_encoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers
        )
        self.classification_head = nn.Linear(d_model, n_classes)

    def generate_positional_encoding(self, d_model, max_len=243):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, : x.size(1)]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        class_logits = self.classification_head(x)
        return x, class_logits

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        z_proto: torch.Tensor,
        z_query: torch.Tensor,
    ) -> torch.Tensor:

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores

def predictClass(matrix_file_path):
    # Load the saved TransformerEncoder model
    saved_model_path = "TE-64_alpha_0.9.pt"

    # Instantiate the TransformerEncoder as the backbone
    n_features = 114
    n_classes = 64
    encoder = TransformerEncoder(n_features=n_features)
    encoder.load_state_dict(torch.load(saved_model_path))
    
    # Load the saved model
    loaded_model = PrototypicalNetworks(encoder)
    loaded_model.load_state_dict(torch.load('trained_model.pt'))
    loaded_model.eval()

    values = []
    num_rows = 0

    with open(matrix_file_path, "r") as f_input:
        reader = csv.reader(f_input)
        for row in reader:
            row_values = []
            for i in range(len(row) - 1):
                column_value = ast.literal_eval(row[i])
                row_values.append(column_value)
            values.append(torch.tensor(row_values))
            num_rows += 1

    
    z_proto = torch.load('best_z_proto.pt')
    print(z_proto.shape)
    
    values = torch.stack(values)
    print(values)
    z_query,_ = loaded_model.backbone.forward(torch.tensor(values, dtype=torch.float32))
    print(z_query.shape)
    
    predicted_class = None

    #z_proto - prototypes from the database
    # Get the prediction scores
    with torch.no_grad():
        scores = loaded_model(z_proto, z_query)
        _, predictions = torch.max(scores, 1)
        predicted_class = predictions[0].item()
        # print(matrix_labels)
        # accuracy = accuracy_score(matrix_labels, predictions.detach().numpy())
        # print(accuracy)

    return {"predicted_class" : predicted_class}


@app.route('/store', methods=['GET'])
def get_stores():
    return {"stores": stores}

@app.route('/test', methods=['GET'])
def test():
    # Usage
    files = get_file_details()

    # Print file details
    for file in files:
      if file['file_name']=='yolov4.cfg' or file['file_name']=='yolov4.weights'  or file['file_name']=='025_003_005.mp4':
         print("File Name:", file['file_name'])
         print("File Path:", file['file_path'])
         print("File Size (bytes):", file['file_size'])
         print("-" * 30)
    return "File view option successful!"

@app.route('/predict', methods=['POST'])
def GenerateSkeletonSeq():
    print("seq gen started")
    
    request_data = request.get_json()
    
    filename=request_data["filename"]
    
    print(filename)
    
    download_file_firestore('t-13859.appspot.com',filename,filename)

    out_filepath_30_fps = convert_30_fps(filename)
    
    # output_path_videos_with_bounding_box=draw_bounding_box(out_filepath_30_fps)

    # coordinates=get_bounding_boxes(output_path_videos_with_bounding_box)
    
    # bb_cor=find_bounding_box(coordinates)
    
    # cropped_video_path = crop_video(output_path_videos_with_bounding_box, (bb_cor[0][0], bb_cor[0][1], bb_cor[1][0], bb_cor[2][1]))
    print(out_filepath_30_fps)
    extract_skeleton_locations(out_filepath_30_fps)
    
    print("locations extracted")

    matrix_creation("skeleton_locations.csv")

    print("matrix created")

    predicted_class = predictClass("padded_matrix_file.csv")

    return predicted_class
    
    



if __name__ == '__main__':
    app.run(debug=false, host='0.0.0.0')