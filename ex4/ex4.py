import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
from multiprocessing import Process
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def stabilize_horizontal_motion(matrix):
    # zero rotation components
    matrix[0, 1] = 0
    matrix[1, 0] = 0
    
    return matrix

def align_images(image1, image2, calc_direction = False):
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
    
    
    def apply_blur(img):
        h, w = img.shape[:2]
        small_size = (max(1, int(w * 0.5)), max(1, int(h * 0.5)))
        blurred = cv2.resize(img, small_size)
        return cv2.resize(blurred, (w, h))
    
    # LK works better with blurred images (good thing I learned to Test 2) so lets blur :)
    gray1 = apply_blur(gray1)
    gray2 = apply_blur(gray2)
    
    # Calculate optical flow (Lucas-Kanade) to find corresponding points in image2
    lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    points2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)
    
    # Select valid points
    points1_valid = points1[st == 1]
    points2_valid = points2[st == 1]
    
    matrix, inliers = cv2.estimateAffinePartial2D(points1_valid, points2_valid, method=cv2.RANSAC, ransacReprojThreshold=1)
    
    matrix = to_homogeneous(matrix)
    
    direction = 'left' # default
    if calc_direction:
        motion_vectors = points2_valid - points1_valid
        dx = motion_vectors[:, 0].mean()
        dy = motion_vectors[:, 1].mean()
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'

    return stabilize_horizontal_motion(matrix), direction


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
    num_frames = len(frames)
    ref_index = num_frames // 2
    transformations = [np.eye(3)]  # Identity matrix for the reference frame

    y_translations = [0]  # Initial Y translation for the reference frame

    # Right-side transformations
    right_transform = np.eye(3)
    for i in range(ref_index + 1, num_frames):
        matrix, _ = align_images(frames[i - 1], frames[i])
        right_transform = right_transform @ matrix
        transformations.append(right_transform)
        y_translations.append(right_transform[1, 2])

    # Left-side transformations
    left_transform = np.eye(3)
    for i in range(ref_index - 1, -1, -1):
        matrix, _ = align_images(frames[i + 1], frames[i])
        left_transform = matrix @ left_transform
        transformations.insert(0, left_transform)
        y_translations.insert(0, left_transform[1, 2])

    # Calculate the median Y translation
    median_y_translation = np.median(y_translations)

    # Adjust all transformations to stabilize around the median Y translation
    for i in range(len(transformations)):
        transformations[i][1, 2] -= median_y_translation

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


def trim_black_borders(image):
    """
    Trims black rows and columns from the given image.
    
    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        
    Returns:
        numpy.ndarray: Cropped image with black borders removed.
    """
    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Find non-black pixels
    non_black_pixels = np.where(gray > 0)

    # Get bounding box of non-black pixels
    y_min, y_max = non_black_pixels[0].min(), non_black_pixels[0].max()
    x_min, x_max = non_black_pixels[1].min(), non_black_pixels[1].max()

    # Crop the image to the bounding box
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    return cropped_image


def stitch_panorama(video_name, frames, transformations, canvas_size, frame_x_offset=0):
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

def rotate_frame(frame, direction):
    """Rotates the frame based on the detected direction."""
    if direction == 'right':
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif direction == 'left':
        return frame  # No rotation needed
    elif direction == 'up':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 'down':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def rotate_frame_back(frame, direction):
    """Rotates the frame back to its original orientation."""
    if direction == 'right':
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif direction == 'left':
        return frame  # No rotation needed
    elif direction == 'up':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 'down':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

def detect_motion_direction(frames):
    # vote with motion of first 5 frames relative to the first frame
    dirction_vote = {
        'left': 0,
        'right': 0,
        'up': 0,
        'down': 0
    }
    
    for i in range(1,6):
        _, direction = align_images(frames[0], frames[i], calc_direction=True)
        dirction_vote[direction] += 1
        
    return max(dirction_vote, key=dirction_vote.get)

def generate_mosaic_video(video_path, output_dir, dynamic = False):
    
    video_dir, video_name = os.path.split(video_path)
    
    logger.info('Generating mosaic video for %s', video_name)
    if dynamic:
        logger.info('dynamic mosaic for: %s', video_path)
    start_time = time.time()
    
    # Extract frames from the video
    frames = extract_frames(video_path)
    
    logger.info('Extracted %d frames from %s', len(frames), video_name)

    # Detect predominant motion direction
    motion_direction = detect_motion_direction(frames)
    logger.info(f"Detected motion direction for {video_name}: {motion_direction}")

    # Rotate frames to make the motion rightward
    frames = [rotate_frame(frame, motion_direction) for frame in frames]

    # Calculate transformations and stitch panorama
    logger.info('Calculating transformations for %s', video_name)
    transformations, ref_index = calculate_transformations(frames)
    logger.info('Calculated transformations DONE for %s', video_name)
    
    logger.info('Calculating canvas size for %s', video_name)
    canvas_size = calculate_canvas_size(frames, transformations, ref_index)
    logger.info('Calculated canvas size DONE for %s', video_name)
    
    # Stitch panorama
    target_frame_count = 30
    total_frames = len(frames)
    step_size = max(total_frames // target_frame_count, 1)
    if not dynamic:
        step_size*=2
        
    selected_indices = range(10, total_frames, step_size)[:target_frame_count]
    
    logger.info('Stitching panorama for %s', video_name)
    
    with ThreadPoolExecutor() as executor:
        panoramas = list(executor.map(
            lambda i: stitch_panorama(video_name, frames, transformations, canvas_size, i),
            selected_indices
        ))
        
                
    if dynamic:
        # trim black borders
        final_panoramas = [trim_black_borders(panorama) for panorama in panoramas]
        # reverse order
        final_panoramas = final_panoramas[::-1]
        
    else:
        panoramas_reverse = panoramas[::-1]
        final_panoramas = panoramas + panoramas_reverse
        
    final_panoramas = [rotate_frame_back(panorama, motion_direction) for panorama in panoramas]
        
    # Save the final video
    images_to_video(final_panoramas, f"{output_dir}/{video_name}", fps=target_frame_count)
    end_time = time.time()
    
    #log time in HH:MM:SS
    logger.info('Generated mosaic video for %s in %s', video_name, time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))


if __name__ == "__main__":
    processes = []
    dynamic_mosaics = ['Trees.mp4', 'Iguazu.mp4']

    # Iterate over all videos in the input directory
    for video in os.listdir("input"):
        if video.endswith(".mp4"):
            input_path = f"input/{video}"
            output_path = "my_output"

            # Create a new process for each video
            process = Process(target=generate_mosaic_video, args=(input_path, output_path, video in dynamic_mosaics))
            processes.append(process)
            process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()