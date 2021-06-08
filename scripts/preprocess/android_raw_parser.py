import sys
sys.path.insert(0, '..')
from utils.utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import cv2
from PIL import Image
import time
from threading import Thread


def save_files(dir, depths, confidences, vid_frame):
    Image.fromarray(depths).save(os.path.join(dir, f'depth_{i}.png'))
    Image.fromarray(confidences).save(os.path.join(dir, f'confidence_{i}.png'))
    Image.fromarray(vid_frame).save(os.path.join(dir, f'rgb_{i}.png'))


def save_image(img, dir, file_name):
    Image.fromarray(img).save(os.path.join(dir, file_name))


def convert_to_rgb(frame: np.ndarray):
    max = frame.max(initial=0)
    min = frame.min(initial=255)
    return np.interp(frame, [min, max], [0, 255]).astype(np.uint8)


def get_frame_count(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def get_video_frame(cap, frame_ind):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
    ret, frame = cap.read()
    return frame

set_path()

HEIGHT = 480
WIDTH = 640
FILTER_DIST = 350

data_dir = './data'
raw_data_dir = os.path.join(data_dir, 'raw', 'android_data')
processed_data_dir = os.path.join(data_dir, 'raw', 'rgbd')
data_files = os.listdir(raw_data_dir)

for run_folder in data_files:
    if run_folder != 's_run_t_3':
        continue

    depth_raw = []
    depth_timestamp = []

    cap = cv2.VideoCapture(os.path.join(raw_data_dir, run_folder, 'video.mp4'))
    print(cap)

    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'video length: {vid_length}')
    print(f'vide fps: {vid_fps}')

    lines = np.memmap(os.path.join(raw_data_dir, run_folder, 'depth.txt'), mode='r', dtype='uint16')
    lines = np.reshape(lines, (-1, WIDTH * HEIGHT))
    timestamps = np.fromfile(os.path.join(raw_data_dir, run_folder, 'time.txt'), np.uint64, sep="\n")
    timestamps = (timestamps - timestamps[0]) / 1000000000

    print('reading raw depth file...')
    for line in lines:
        # last character is a '\n', so don't include it in the data
        depth_raw.append(line)

    depth_raw = np.array(depth_raw)

    num_frames = depth_raw.shape[0]
    # num_frames = 1

    print('reading raw depth file complete')
    print(f'number of frames: {num_frames}')

    print('parsing depth information...')
    frames = []
    confidence_mapper = lambda confidence_raw:  1 if confidence_raw == 0 else confidence_raw/7.0
    confidence_vectorizer = np.vectorize(confidence_mapper)

    for i in range(int(num_frames)):
        print(f'processing frame {i}/{num_frames}')
        depths = np.bitwise_and(depth_raw[i], 0x1FFF)
        confidences = np.right_shift(depth_raw[i], 13)
        confidences = confidence_vectorizer(confidences)
        frames.append([depths, confidences])

    print('parsing depth information complete')

    processed_folder = os.path.join(processed_data_dir, run_folder)
    mkdir(processed_folder)

    video_writer = cv2.VideoWriter(os.path.join(processed_folder, 'video.avi'),
                                   cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1320, 640))

    print('rendering graphs...')
    time_start = time.time()
    for i, frame in enumerate(frames):
        frame_offset = 5

        print(f'rendering frame {i}/{num_frames}')

        time1 = time.time()
        depth_pct = i / num_frames
        # vid_frame_ind = round(depth_pct * vid_length)

        vid_frame_ind = round(timestamps[i] * vid_fps) - frame_offset
        if vid_frame_ind < 0:
            continue

        if vid_frame_ind >= vid_length:
            break

        print(f'video frame ind: {vid_frame_ind} / {vid_length}')

        vid_frame = get_video_frame(cap, vid_frame_ind)
        vid_frame = cv2.resize(vid_frame, (640, int(640 / 1920 * 1080)))

        time2 = time.time()

        depths = np.reshape(np.array(frame[0]), (HEIGHT, -1))
        depths[depths > FILTER_DIST] = 0

        mask = np.zeros(depths.shape, dtype='uint8')
        mask[depths == 0] = 255

        confidences = np.reshape(np.array(frame[1]), (HEIGHT, -1))
        confidence_rgb = convert_to_rgb(confidences.copy())

        # confidence_rgb = mask

        confidence_rgb = np.stack((confidence_rgb,)*3, axis=-1)

        depths_rgb = convert_to_rgb(depths.copy())
        depths_rgb = np.stack((depths_rgb,)*3, axis=-1)

        final_frame = np.zeros((vid_frame.shape[0] + 480 * 2, 640, 3))

        final_frame[:vid_frame.shape[0], :, :] = vid_frame
        final_frame[vid_frame.shape[0]:vid_frame.shape[0] + confidence_rgb.shape[0], :, :] = confidence_rgb
        final_frame[vid_frame.shape[0] + confidence_rgb.shape[0]:, :, :] = depths_rgb

        # cv2.imshow('rgb', cv2.rotate(vid_frame, cv2.ROTATE_90_CLOCKWISE))
        # cv2.imshow('confidence', cv2.rotate(confidence_rgb, cv2.ROTATE_90_CLOCKWISE))
        cv2.imshow('final', cv2.rotate(final_frame, cv2.ROTATE_90_CLOCKWISE).astype(np.uint8))
        video_writer.write(cv2.rotate(final_frame, cv2.ROTATE_90_CLOCKWISE).astype(np.uint8))
        cv2.waitKey(1)

        time3 = time.time()

        Thread(target=save_files, args=(processed_folder, depths, confidence_rgb,
                                        cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB).astype(np.uint8))).start()

        time4 = time.time()

        plt.subplot(211), plt.imshow(cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB))
        plt.subplot(223), plt.imshow(depths), plt.title(run_folder + ': depth')
        plt.subplot(224), plt.imshow(confidences), plt.title(run_folder + ': confidence')
        # plt.show()
        # plt.pause(0.001)

        time5 = time.time()

        dur_1 = time2 - time1
        dur_2 = time3 - time2
        dur_3 = time4 - time3
        dur_4 = time5 - time4

        total_time = dur_1 + dur_2 + dur_3 + dur_4

        print('getting video frame:', time2 - time1)
        print('cleaning depth data:', time3 - time2)
        print('saving to file:', time4 - time3)
        print('plotting:', time5 - time4)
        print()

    video_writer.release()
    duration_all = time.time() - time_start
    print(duration_all)

    # plt.show()
