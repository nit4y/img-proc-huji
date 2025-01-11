import cv2
import numpy as np
from math import ceil, floor
import mediapy as media

def align_images(image1, image2, isHomography = False):
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
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
    points1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    # Calculate optical flow (Lucas-Kanade) to find corresponding points in image2
    lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    points2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)
    
    # Select valid points
    points1_valid = points1[st == 1]
    points2_valid = points2[st == 1]
    
    if isHomography:
        matrix, inliers = cv2.findHomography(points1_valid, points2_valid, cv2.RANSAC, 5.0)
        return matrix
    
    # Estimate a transformation matrix (Affine or Homography)
    # For Affine transformation (3x2 matrix):
    
    matrix, inliers = cv2.estimateAffinePartial2D(points1_valid, points2_valid)
    
    return to_homogeneous(matrix)


def to_homogeneous(affine_matrix):
    """Converts a 3x2 affine matrix to a 3x3 homogeneous matrix."""
    return np.vstack([affine_matrix, [0, 0, 1]])


# Example usage
def img_test():
    # Load consecutive frames
    image1 = cv2.imread("tarantino.jpg")
    image2 = cv2.imread("tarantino.jpg")
    
    # Align images
    transformation_matrix, aligned_image = align_images(image1, image2)
    
    # Display results
    print("Transformation Matrix:")
    print(transformation_matrix)
    
    cv2.imshow("Image1 (Reference)", image1)
    cv2.imshow("Image2 (Original)", image2)
    cv2.imshow("Image2 (Aligned)", aligned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    """Calculates transformations from each frame to the next."""
    num_frames = len(frames)
    ref_index = num_frames // 2
    transformations = [np.eye(3)] # Start with an identity matrix for the first frame
    
    # calculate cumulative transformations
    right_transform = transformations[0]
    left_transform = transformations[0]
    count = 0
    for i in range(ref_index, num_frames):
        right_matrix = align_images(frames[i - 1], frames[i])
        right_transform = right_transform @ right_matrix
        transformations.append(right_transform)
        
        left_matrix = align_images(frames[ref_index - count], frames[ref_index - count - 1])
        left_transform = left_matrix @ left_transform
        # insert left transformation at the beginning of the list
        transformations.insert(0, left_transform)
        count+=1
            
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

def stitch_panorama(frames, transformations, canvas_size, ref_index):
    """Stitches frames into a panoramic mosaic."""
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    offset_x, offset_y = canvas_size[2], canvas_size[3]
    
    prev_trans = None
    for i, frame in enumerate(frames):
        
        transformation = np.linalg.inv(transformations[i])
        transformation[0, 2] += offset_x
        transformation[1, 2] += offset_y
        if i != 0:
            frame = frame[:, :abs(ceil(prev_trans[0, 2]) - ceil(transformation[0, 2]))]
        prev_trans = transformation
        warped_frame = cv2.warpPerspective(frame, transformation, (canvas.shape[1], canvas.shape[0]))
        mask = (warped_frame.sum(axis=2) > 0).astype(np.uint8)
        canvas[mask == 1] = warped_frame[mask == 1]

    return canvas

def main(video_path):
    frames = extract_frames(video_path)
    transformations, ref_index = calculate_transformations(frames)
    canvas_size = calculate_canvas_size(frames, transformations, ref_index)
    panorama = stitch_panorama(frames, transformations, canvas_size, ref_index)
    
    cv2.imwrite("panorama.jpg", panorama)
    # cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("input/boat.mp4")