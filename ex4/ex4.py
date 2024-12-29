import cv2
import numpy as np
import mediapy as media

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
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
    points1 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    
    # Calculate optical flow (Lucas-Kanade) to find corresponding points in image2
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    points2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None, **lk_params)
    
    # Select valid points
    points1_valid = points1[st == 1]
    points2_valid = points2[st == 1]
    
    # Estimate a transformation matrix (Affine or Homography)
    # For Affine transformation (3x2 matrix):
    matrix, inliers = cv2.estimateAffinePartial2D(points1_valid, points2_valid)
    
    # If you want a Homography (3x3 matrix) instead:
    # matrix, inliers = cv2.findHomography(points1_valid, points2_valid, cv2.RANSAC, 5.0)
    
    # Warp image2 to align with image1
    height, width = image1.shape[:2]
    aligned_image = cv2.warpAffine(image2, matrix, (width, height))
    
    return matrix, aligned_image

# Example usage
if __name__ == "__main__":
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
