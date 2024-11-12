import numpy as np
from scipy.spatial.distance import euclidean
import mediapy as media
from PIL import Image

def threshhold_factory(video_type: int) -> int:
    if video_type == 1:
        return 10
    elif video_type == 2:
        return 30
    return 0

def main_cumsum(video_path: str, video_type: int) -> tuple:
    # Parameters
    previous_hist = None
    max_diff = 0
    max_frame = None
    max_frame_index = -1

    for frame_index, frame in enumerate(media.read_video(video_path)):
        
        gray_frame = Image.fromarray(frame).convert('L')
        gray_frame = np.array(gray_frame) 

        equalized_hist, _ = np.histogram(gray_frame, bins=256, range=(0, 255), density=True)

        if previous_hist is not None:
            cum_sum_prev = np.cumsum(previous_hist)
            cum_sum_current = np.cumsum(equalized_hist)

            dist = euclidean(cum_sum_prev, cum_sum_current)

            if dist > max_diff:
                max_diff = dist
                max_frame = frame
                max_frame_index = frame_index

        # update the previous histogram with the current equalized histogram
        previous_hist = equalized_hist

    return max_frame_index, max_frame, max_diff

if __name__ == "__main__":
    max_frame_index, max_diff_frame, max_diff = main_cumsum("./video3_category2.mp4", 2)
    print("Cut in frame:", max_frame_index)
    print("With diff:", max_diff)
