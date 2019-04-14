import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def trimmer():
    start_time = 126
    end_time = 147
    ffmpeg_extract_subclip("../../data/videos/case-3-2D.avi", start_time, end_time,
                           targetname="../../data/videos/clip-case-3.mp4")


if __name__ == "__main__":
    trimmer()
