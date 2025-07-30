import os
from tqdm import tqdm
import cv2
import numpy as np
import sys

# Function to calculate optical flow score for a video

def cal_opticalflow_score(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read video")
        return 0.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_magnitude_list  = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # 检查光流向量的平均大小
        avg_flow_magnitude = np.mean(flow_magnitude)
        # print(f"Average flow magnitude: {avg_flow_magnitude}")
        flow_magnitude_list.append(avg_flow_magnitude)

        prev_gray = gray
    cap.release()
    return np.mean(flow_magnitude_list) 

# Function to compute the average optical flow score for all videos in a folder
def calculate_average_score(video_folder):
    total_score = 0
    video_count = 0

    # List all files in the given folder
    for video_file in tqdm(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, video_file)
        
        # Check if the file is a video (you might need to adjust the extension checks as per your requirements)
        if video_path.endswith(('.mp4', '.avi', '.mov')):
            # Call the function to get the optical flow score
            score = cal_opticalflow_score(video_path)
            total_score += score
            video_count += 1

    # Avoid division by zero
    if video_count == 0:
        return 0
    
    # Calculate the average score
    average_score = total_score / video_count
    return average_score

# Specify the path to your video folder here
video_folder = sys.argv[1]

# Calculate and print the average optical flow score
average_score = calculate_average_score(video_folder)
print("Average Optical Flow Score:", average_score)
print("==========="* 10)