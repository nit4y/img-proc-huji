import numpy as np
from scipy.spatial.distance import euclidean
import mediapy as media
from PIL import Image

def main(video_path: str, video_type: int) -> tuple[int]:
    
    previous_hist = None
    max_dist = -1
    cuts = tuple()

    for frame_index, frame in enumerate(media.read_video(video_path)):
        
        gray_frame = Image.fromarray(frame).convert('L')
        gray_frame = np.array(gray_frame) 
        
        hist, _ = np.histogram(gray_frame, bins=256, range=(0, 255), density=True)

        if previous_hist is not None:
            cum_sum_prev = np.cumsum(previous_hist)
            cum_sum_current = np.cumsum(hist)

            dist = euclidean(cum_sum_prev, cum_sum_current)

            if dist > max_dist:
                max_dist = dist
                cuts = (frame_index-1, frame_index)
                
        # update the previous histogram with the current equalized histogram
        previous_hist = hist

    return cuts
