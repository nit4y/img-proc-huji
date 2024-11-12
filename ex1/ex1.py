import numpy as np
from scipy.spatial.distance import euclidean
import mediapy as media
from PIL import Image

def threshhold_factory(video_type: int) -> int:
    if video_type == 1:
        return 1
    elif video_type == 2:
        return 1.5
    return 0

def main(video_path: str, video_type: int) -> tuple:
    
    previous_hist = None
    threshhold = threshhold_factory(video_type)
    cuts = []

    for frame_index, frame in enumerate(media.read_video(video_path)):
        
        gray_frame = Image.fromarray(frame).convert('L')
        gray_frame = np.array(gray_frame) 
        
        equalized_hist, _ = np.histogram(gray_frame, bins=256, range=(0, 255), density=True)

        if previous_hist is not None:
            cum_sum_prev = np.cumsum(previous_hist)
            cum_sum_current = np.cumsum(equalized_hist)

            dist = euclidean(cum_sum_prev, cum_sum_current)

            if dist > threshhold:
                cuts.append(frame_index)
                
        # update the previous histogram with the current equalized histogram
        previous_hist = equalized_hist

    return tuple(cuts)

if __name__ == "__main__":
    cuts = main("./video3_category2.mp4", 1)
    print("Cuts in frames: ", cuts)
