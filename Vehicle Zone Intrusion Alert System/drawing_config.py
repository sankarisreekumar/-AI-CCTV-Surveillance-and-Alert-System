# drawing_config.py

def get_lane_points(video_path):
    """
    Return lane lines based on video or camera
    """

    # Example: different configs for different sources
    if "camera1" in video_path:
        left_line = ((39, 474), (192, 213))
        right_line = ((281, 402), (308, 194))

    elif "camera2" in video_path:
        left_line = ((50, 480), (200, 200))
        right_line = ((300, 400), (350, 180))

    else:
        # Default
        left_line = ((39, 474), (192, 213))
        right_line = ((281, 402), (308, 194))

    return left_line, right_line