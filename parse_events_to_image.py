import time
import numpy as np
import cv2
import bisect
import matplotlib.pyplot as plt
import os

class Events():
    def __init__(self, events_file):
        self.t, self.x, self.y, self.p = self.extract_event_data(events_file)

    def extract_event_data(self, events_file):
        t, x, y, p = [], [], [], []
        fileio = open(events_file, 'r')
        for line in fileio:
            words = line.split(" ")
            t.append(float(words[0]))
            x.append(int(words[1]))
            y.append(int(words[2]))
            p.append(int(words[3]))
        fileio.close()
        return t, x, y, p


class Images():
    def __init__(self, images_file):
        self.t, self.frames_path, self.frames = self.extract_image_data(images_file)

    def extract_image_data(self, images_file):
        t, frames_path, frames = [], [], []
        fileio = open(images_file, 'r')
        for line in fileio:
            words = line.split(" ")
            t.append(float(words[0]))
            frames_path.append(words[1].strip())
            frames.append(None)
        fileio.close()
        return t, frames_path, frames


class parse_event_to_frame():
    def __init__(self, events_file, images_file):
        self.events = Events(events_file)
        self.images = Images(images_file)
        self.delta_t = 0.038865

        self.w = 240
        self.h = 180

    def make_event_image(self, start_time, end_time):
        if start_time < self.events.t[0] or end_time > self.events.t[-1]:
            print(f"Invalid Start/End Time Provided: {start_time} to {end_time}")
            return None

        pos_events = np.zeros((self.h, self.w))
        neg_events = np.zeros((self.h, self.w))

        start_idx = bisect.bisect_left(self.events.t, start_time)
        end_idx = bisect.bisect_right(self.events.t, end_time)

        for idx in range(start_idx, end_idx):
            x, y, p = self.events.x[idx], self.events.y[idx], self.events.p[idx]
            if p == 1:
                pos_events[y, x] += 1
            else:
                neg_events[y, x] += 1

        event_img = pos_events - neg_events

        if np.max(np.abs(event_img)) > 0:
            event_img = np.sign(event_img) * np.log1p(np.abs(event_img))
            event_img = ((event_img - np.min(event_img)) * 255 /
                         (np.max(event_img) - np.min(event_img)))

        event_img = event_img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        event_img = clahe.apply(event_img)

        return event_img

    def process_all_frames(self):
        all_frames = []
        for t in self.images.t:
            event_img = self.make_event_image(t - self.delta_t, t)
            if event_img is not None:
                all_frames.append(event_img)
        return all_frames

    def stitch_images(self, frames):
        stitcher = cv2.Stitcher_create()
        status, stitched_img = stitcher.stitch(frames)
        if status == cv2.Stitcher_OK:
            return stitched_img
        else:
            print("Error in stitching images")
            return None


def create_video_from_images(image_folder, output_video_path, frame_rate=30):
    images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images.sort()

    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        out.write(img)

    out.release()
    print(f"video saved in: {output_video_path}")


# def visualize_frames_in_groups(frames, frames_per_plot=9):
#     total_frames = len(frames)
#     if total_frames == 0:
#         print("No frames to display")
#         return
#
#     num_groups = (total_frames + frames_per_plot - 1) // frames_per_plot
#
#     for group in range(num_groups):
#         start_idx = group * frames_per_plot
#         end_idx = min((group + 1) * frames_per_plot, total_frames)
#         current_frames = frames[start_idx:end_idx]
#
#         cols = min(3, len(current_frames))
#         rows = (len(current_frames) + cols - 1) // cols
#
#         plt.figure(figsize=(15, 5 * rows))
#         for i, frame in enumerate(current_frames):
#             plt.subplot(rows, cols, i + 1)
#             plt.imshow(frame, cmap='gray')
#             plt.title(f'Frame {start_idx + i + 1}')
#             plt.axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#         # הוספת השהייה קצרה בין קבוצות התמונות
#         plt.pause(0.1)


def save_frames(frames, output_dir="output_frames"):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        filename = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(filename, frame)
    print(f"Saved {len(frames)} frames to {output_dir}/")

def create_videos():
    folder_name = "videos"
    try:
        os.mkdir(folder_name)
        print(f"Created folder: {folder_name}")
    except FileExistsError:
        print(f"The folder '{folder_name}' already exists.")

    image_folder = r"..\Final Project\output_frames"
    event_camera_video = r"..\Final Project\videos\event_camera_video.mp4"
    create_video_from_images(image_folder, event_camera_video)

    image_folder = r"..\Final Project\images"
    images_video = r"..\Final Project\videos\images_video.mp4"
    create_video_from_images(image_folder, images_video)

    # sharp video
    image_folder = r"..\Final Project\enhanced_frames"
    images_video = r"..\Final Project\videos\enhanced_frames_video.mp4"
    create_video_from_images(image_folder, images_video)

    merged_video_path = r"..\Final Project\videos\merged_video.mp4"
    return images_video, event_camera_video, merged_video_path


def merge_videos(video_path1, video_path2, output_video_path, frame_rate=30):
    video1 = cv2.VideoCapture(video_path1)
    video2 = cv2.VideoCapture(video_path2)

    width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width1 != width2 or height1 != height2:
        new_width = min(width1, width2)
        new_height = int(new_width * height1 / width1)
        resize_video = lambda vid: cv2.resize(vid, (new_width, new_height))
    else:
        new_width = width1
        new_height = height1
        resize_video = lambda vid: vid

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (new_width * 2, new_height))

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if not ret1 or not ret2:
            break

        frame1 = resize_video(frame1)
        frame2 = resize_video(frame2)
        combined_frame = cv2.hconcat([frame1, frame2])
        out.write(combined_frame)

    video1.release()
    video2.release()
    out.release()

    print(f"merged video saved in: {output_video_path}")


SIGMA = 1.5
STRENGTH = 1.5


def unsharp_mask(image_path, output_dir, file_name, sigma=1.5, strength=1.5):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    output_path = os.path.join(output_dir, f"{file_name}_CV_{sigma}_{strength}.png")
    cv2.imwrite(output_path, sharpened)
    return output_path


def check_folder(folder_name):
    folder_name = "enhanced_frames"
    try:
        os.mkdir(folder_name)
        print(f"Created folder: {folder_name}")
    except FileExistsError:
        print(f"The folder '{folder_name}' already exists.")


def sharp_frames():
    check_folder("output_frames")
    check_folder("enhanced_frames")

    input_dir = r"..\Final Project\output_frames"
    output_dir = r"..\Final Project\enhanced_frames"

    start = time.time()
    images = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    for file_name in images:
        input_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        unsharp_mask(input_path, output_dir, base_name, SIGMA, STRENGTH)

    end = time.time()

    print(f"Processed {len(images)} images with OpenCV unsharp mask.")
    print(f"Total time: {end - start:.2f} seconds")




if __name__ == "__main__":
    start = time.time()
    print("Processing events...")
    dataset = parse_event_to_frame(r"./events.txt", r"./images.txt")
    print("Processing frames...")
    frames = dataset.process_all_frames()

    if frames:
        print(f"Generated {len(frames)} frames")
        save_frames(frames)
    else:
        print("No frames were generated")

    print("Sharp frames...")
    sharp_frames()
    print("Creating videos...")

    images_video, event_camera_video, merged_video_path = create_videos()
    merge_videos(images_video, event_camera_video,merged_video_path)

    end = time.time()
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"{str(minutes).zfill(2)}:{str(seconds).zfill(2)}")
