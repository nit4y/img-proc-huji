import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

def stabilize_horizontal_motion(matrix):
    # zero rotation components
    matrix[0, 1] = 0
    matrix[1, 0] = 0
    
    return matrix


def align_images(image1, image2):
    """
    Aligns image2 to image1 using the Lucas-Kanade optical flow method.
    
    Parameters:
        image1 (numpy.ndarray): First image (reference frame).
        image2 (numpy.ndarray): Second image (to be aligned).
    
    Returns:
        numpy.ndarray: Transformation matrix.
        numpy.ndarray: Aligned version of image2.
    """
    # Convert images to grayscale for LK
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Detect good features to track in image1
    harris1 = cv2.cornerHarris(gray1, blockSize=7, ksize=9, k=0.05)
    harris1 = cv2.dilate(harris1, None)
    points1 = np.argwhere(harris1 > 0.01 * harris1.max())
    points1 = np.expand_dims(points1[:, [1, 0]], axis=1).astype(np.float32)
        
    # Calculate optical flow (Lucas-Kanade) to find corresponding points in image2
    lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    points2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)
    
    # Select valid points
    points1_valid = points1[st == 1]
    points2_valid = points2[st == 1]
    
    matrix, inliers = cv2.estimateAffinePartial2D(points1_valid, points2_valid, method=cv2.RANSAC)
    
    matrix =  to_homogeneous(matrix)
    
    return stabilize_horizontal_motion(matrix)


def to_homogeneous(affine_matrix):
    """Converts a 3x2 affine matrix to a 3x3 homogeneous matrix."""
    return np.vstack([affine_matrix, [0, 0, 1]])

def extract_frames(video_path):
    """Extracts frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def calculate_transformations(frames):
    """Calculates stabilized transformations from each frame to the next."""
    num_frames = len(frames)
    ref_index = num_frames // 2
    transformations = [np.eye(3)]  # identity matrix for the reference frame

    # right-side transformations
    right_transform = np.eye(3)
    for i in range(ref_index + 1, num_frames):
        matrix = align_images(frames[i - 1], frames[i])
        right_transform = right_transform @ matrix
        transformations.append(right_transform)

    # left-side transformations
    left_transform = np.eye(3)
    for i in range(ref_index - 1, -1, -1):
        matrix = align_images(frames[i + 1], frames[i])
        left_transform = matrix @ left_transform
        transformations.insert(0, left_transform)

    return transformations, ref_index

def calculate_canvas_size(frames, transformations, ref_index):
    """Calculates the final canvas size."""
    min_x, min_y = 0, 0
    max_x, max_y = frames[ref_index].shape[1], frames[ref_index].shape[0]

    for i in range(len(frames)):
        h, w = frames[i].shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])

        transformed_corners = [np.dot(transformations[i], corner) for corner in corners]
        transformed_corners = np.array(transformed_corners)
        min_x = min(min_x, transformed_corners[:, 0].min())
        min_y = min(min_y, transformed_corners[:, 1].min())
        max_x = max(max_x, transformed_corners[:, 0].max())
        max_y = max(max_y, transformed_corners[:, 1].max())
        
    return int(max_x - min_x), int(max_y - min_y), -int(min_x), -int(min_y)

def stitch_panorama(frames, transformations, canvas_size, frame_x_offset=0):
    """takes columns from frames into a panoramic mosaic."""
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    offset_x, offset_y = canvas_size[2], canvas_size[3]

    prev_leftmost_x = 0
    prev_warped_frame = []

    for i, frame in enumerate(frames):
        transformation = np.linalg.inv(transformations[i])
        transformation[0, 2] += offset_x
        transformation[1, 2] += offset_y
        
        # Warp the frame to the canvas
        warped_frame = cv2.warpPerspective(frame, transformation, (canvas.shape[1], canvas.shape[0]))

        # Initialize curr_leftmost_x for the first frame
        curr_leftmost_x = 0

        # Find the leftmost non-black pixel in the current warped frame
        non_black_pixels = np.where(warped_frame.sum(axis=2) > 0)
        if non_black_pixels[1].size > 0:
            curr_leftmost_x = np.min(non_black_pixels[1])
            
        if len(prev_warped_frame) > 0:
            # Create a mask with the same size as the canvas
            mask = (prev_warped_frame.sum(axis=2) > 0).astype(np.uint8)

            # Keep only the first x_movement columns in the mask
            # Keep only the columns between prev_leftmost_x and curr_leftmost_x
            mask[:, :prev_leftmost_x+frame_x_offset] = 0  # Zero out columns before prev_leftmost_x
            mask[:, curr_leftmost_x+frame_x_offset:] = 0  # Zero out columns after curr_leftmost_x
            
            # apply mask
            canvas[mask == 1] = prev_warped_frame[mask == 1]

        prev_leftmost_x = curr_leftmost_x
        prev_warped_frame = warped_frame
        
    return canvas


def generate_mosaic_video(video_path, output_dir):
    
    video_dir, video_name = os.path.split(video_path)
    
    frames = extract_frames(video_path)
    transformations, ref_index = calculate_transformations(frames)
    canvas_size = calculate_canvas_size(frames, transformations, ref_index)

    target_frame_count = 30
    total_frames = len(frames)

    # Calculate the step size to evenly distribute 30 frames
    step_size = max(total_frames // target_frame_count * 2, 1)

    # Select frame indices to process
    # 10 is minimum to avoid black columns in the beginning
    selected_indices = range(10, total_frames, step_size)[:target_frame_count]

    with ThreadPoolExecutor() as executor:
        panoramas = list(executor.map(
            lambda i: stitch_panorama(frames, transformations, canvas_size, i),
            selected_indices
        ))
        
    panoramas_reverse = panoramas[::-1]

    # Combine forward and reverse panoramas
    final_panoramas = panoramas + panoramas_reverse

    # Create an MP4 video from the panoramas
    images_to_video(final_panoramas, f"{output_dir}/{video_name}", fps=30)

def images_to_video(images, output_path, fps=30):
    """
    Converts a list of images to an MP4 video.

    Parameters:
        images (list of numpy.ndarray): List of images (frames) to include in the video.
        output_path (str): Path to save the output video file.
        fps (int): Frames per second for the video.
    """
    if not images:
        raise ValueError("No images provided to create the video.")

    height, width, layers = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        out.write(img)

    out.release()
    print(f"Video saved at {output_path}")

if __name__ == "__main__":
    # iterate over all videos in the input directory
    for video in os.listdir("input"):
        if video.endswith(".mp4"):
            generate_mosaic_video(f"input/{video}", "my_output")
    