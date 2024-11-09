import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean  # Corrected import

def threshhold_factory(video_type: int) -> int:

    if video_type == 1:
        return 10
    elif video_type == 2:
        return 30
    
    return 0

    

def main_cumsum(video_path: str, video_type: int) -> tuple:
    # Parameters
    distance_threshold = threshhold_factory(video_type)  # Set the threshold distance for histogram difference
    output_frames_dir = 'frames'

    # Create the frames directory if it doesn't exist
    os.makedirs(output_frames_dir, exist_ok=True)

    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return ()

    # Initialize variables
    previous_hist = None
    significant_frames = []

    # Frame processing loop
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save the original frame
        frame_path = os.path.join(output_frames_dir, f'frame_{frame_index}.png')
        cv2.imwrite(frame_path, frame)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the histogram and normalize it
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Apply histogram equalization
        equalized_frame = cv2.equalizeHist(gray_frame)

        # Compute the equalized histogram
        equalized_hist = cv2.calcHist([equalized_frame], [0], None, [256], [0, 256])
        equalized_hist = cv2.normalize(equalized_hist, equalized_hist).flatten()

        # Compare the cumulative sum of histograms between frames
        if previous_hist is not None:
            cum_sum_prev = np.cumsum(previous_hist)
            cum_sum_current = np.cumsum(equalized_hist)
            
            # Calculate the distance between cumulative sums
            dist = euclidean(cum_sum_prev, cum_sum_current)

            # Check if the distance exceeds the threshold
            if dist > distance_threshold:
                significant_frames.append((frame_index, dist))

        # Update the previous histogram with the current equalized histogram
        previous_hist = equalized_hist
        frame_index += 1

    cap.release()

    # Return the tuple with significant frames and distances
    return tuple(significant_frames)


if __name__ == "__main__":
    # # Get command-line arguments for video_path and video_type
    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <video_path> <video_type>")
    #     sys.exit(1)

    # video_path = sys.argv[1]
    # video_type = sys.argv[2]
    significant_frames = main_cumsum("./video3_category2.mp4", 2)

    print("Significant frames and distances:", significant_frames)
