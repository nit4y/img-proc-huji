import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def fourier_domain_blending(image1, image2, output_filename="hybrid_merged.png"):
    """
    Blend two images in the Fourier domain and save the resulting hybrid image.

    Parameters:
    - image1: First greyscale image as a NumPy array.
    - image2: Second greyscale image as a NumPy array.
    - output_filename: Filename for saving the result image.
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    fft1 = np.fft.fft2(image1)
    fft2 = np.fft.fft2(image2)

    fft1_shifted = np.fft.fftshift(fft1)
    fft2_shifted = np.fft.fftshift(fft2)

    gaussian = build_gaussian_mask(image1)

    gaussian_image = (gaussian * 255).astype(np.uint8)
    cv2.imwrite("q2_gaussian_mask.png", gaussian_image)

    blended_fft1 = fft1_shifted * gaussian
    blended_fft2 = fft2_shifted * (1 - gaussian)

    combined_fft = blended_fft1 + blended_fft2

    combined_fft_ishift = np.fft.ifftshift(combined_fft)
    blended_image = np.fft.ifft2(combined_fft_ishift)

    blended_image = np.abs(blended_image)

    blended_image = cv2.normalize(blended_image, None, 0, 255, cv2.NORM_MINMAX)
    blended_image = blended_image.astype(np.uint8)
    cv2.imwrite(f"q2_{output_filename}", blended_image)

def build_gaussian_mask(image1):

    # in order to build a gaussian mask in the size of the original image
    # we will use numpy's meshgrid

    rows, cols = image1.shape
    x, y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    d = np.sqrt(x**2 + y**2)
    sigma = 0.2
    gaussian = np.exp(-((d**2) / (sigma**2))) * 255

    gaussian = gaussian / np.max(gaussian)
    return gaussian

def q2():
    img1 = cv2.imread("britney.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("tarantino.jpg", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both input images could not be loaded. Make sure the paths are correct.")

    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0])))
        img2 = cv2.resize(img2, (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0])))

    fourier_domain_blending(img1, img2)

    print("Blended image saved as hybrid_merged.png")

def build_laplacian_pyramid(image, levels):
    """
    Create a Laplacian pyramid from an input image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        levels (int): Number of levels in the pyramid.

    Returns:
        list: A list of NumPy arrays representing the Laplacian pyramid.
    """
    # Ensure the image is a floating-point array for precision
    image = image.astype(np.float32)

    # Create the Gaussian pyramid
    gaussian_pyramid = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)

    # Create the Laplacian pyramid
    laplacian_pyramid = []
    for i in range(levels):
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=gaussian_pyramid[i].shape[:2][::-1])
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)

    # Add the smallest level of the Gaussian pyramid to the Laplacian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid

def build_gaussian_pyramid(image, levels):
    gaussian_pyramid = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        if image.shape[0] % 2 != 0 or image.shape[1] % 2 != 0:
            image = cv2.resize(image, ((image.shape[1] // 2) * 2, (image.shape[0] // 2) * 2))
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def pyramid_blending(image_a, image_b, mask, levels=4):
    """
    Perform pyramid blending for two images A and B using a binary mask M.

    Parameters:
    - image_a: First image as a NumPy array.
    - image_b: Second image as a NumPy array.
    - mask: Binary mask as a NumPy array.

    Returns:
    - Blended image as a NumPy array.
    """
    # Ensure inputs have the same dimensions
    if image_a.shape != image_b.shape or image_a.shape[:2] != mask.shape[:2]:
        raise ValueError("Input images and mask must have the same dimensions.")

    # Without converting to float, values will round before stretching
    image_a = image_a.astype(np.float32)
    image_b = image_b.astype(np.float32)

    # Convert mask to float if not already
    mask = mask.astype(np.float32) / 255 if mask.max() > 1 else mask.astype(np.float32)

    # Build pyramids
    laplacian_a = build_laplacian_pyramid(image_a, levels)
    laplacian_b = build_laplacian_pyramid(image_b, levels)
    gaussian_mask = build_gaussian_pyramid(mask, levels)

    # Ensure mask dimensions match pyramid levels
    for i in range(len(gaussian_mask)):
        if gaussian_mask[i].shape[:2] != laplacian_a[i].shape[:2]:
            gaussian_mask[i] = cv2.resize(gaussian_mask[i], (laplacian_a[i].shape[1], laplacian_a[i].shape[0]))

    # Blend pyramids
    laplacian_c = []
    for La, Lb, Gm in zip(laplacian_a, laplacian_b, gaussian_mask):
        Gm = Gm[:, :, np.newaxis] if len(La.shape) == 3 and Gm.ndim == 2 else Gm
        Lc = Gm * La + (1 - Gm) * Lb
        laplacian_c.append(Lc)

    # Reconstruct blended image
    blended_image = laplacian_c[-1]
    for i in range(len(laplacian_c) - 2, -1, -1):
        blended_image = cv2.pyrUp(blended_image, dstsize=laplacian_c[i].shape[:2][::-1])
        blended_image = cv2.add(blended_image, laplacian_c[i])

    return np.clip(blended_image, 0, 255).astype(np.uint8)

def create_binary_half_half_mask(width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, :width // 2] = 1
    return Image.fromarray(mask * 255)

def plot_gaussian_pyramid(image_path, levels=4):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    # Build Gaussian Pyramid
    gaussian_pyramid = build_gaussian_pyramid(img, levels)

    # Plot Gaussian Pyramid
    plt.figure(figsize=(12, 8))
    for i, layer in enumerate(gaussian_pyramid):
        plt.subplot(1, levels + 1, i + 1)
        plt.imshow(layer, cmap='gray')
        plt.title(f"Level {i}")
        plt.axis('off')
    plt.suptitle("Gaussian Pyramid")
    plt.show()

def plot_laplacian_pyramid(image_path, levels=4):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")

    # Build Laplacian Pyramid
    laplacian_pyramid = build_laplacian_pyramid(img, levels)

    # Plot Laplacian Pyramid
    plt.figure(figsize=(12, 8))
    for i, layer in enumerate(laplacian_pyramid):
        plt.subplot(1, levels + 1, i + 1)
        plt.imshow(layer, cmap='gray')
        plt.title(f"Level {i}")
        plt.axis('off')
    plt.suptitle("Laplacian Pyramid")
    plt.show()

def q1():
    # Load input images and mask
    img_a = cv2.imread("britney.jpg")
    img_b = cv2.imread("tarantino.jpg")

    mask = create_binary_half_half_mask(img_a.shape[1], img_a.shape[0])
    mask.save("q1_mask.png")
    mask = cv2.imread("q1_mask.png", cv2.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None or mask is None:
        raise FileNotFoundError("One or more input files could not be loaded. Make sure the paths are correct.")

    # Resize images and mask to match dimensions
    if img_a.shape != img_b.shape:
        img_a = cv2.resize(img_a, (min(img_a.shape[1], img_b.shape[1]), min(img_a.shape[0], img_b.shape[0])))
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    mask = cv2.resize(mask, (img_a.shape[1], img_a.shape[0]))

    # Perform pyramid blending
    blended_result = pyramid_blending(img_a, img_b, mask, 4)

    # Save the blended image
    filename = "q1_blended_image.png"
    cv2.imwrite(filename, blended_result)
    print(f"Blended image saved as {filename}")

if __name__ == "__main__":
    q1()
    q2()

    try:
        plot_gaussian_pyramid("tarantino.jpg")
        plot_laplacian_pyramid("tarantino.jpg")
    except FileNotFoundError as e:
        print(e)
