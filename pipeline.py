#Liberaries
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import re
import datetime
import shutil


#Key frames extraction
#CELL 2

def keyframes_optical_flow(video_path, output_dir, threshold=0.2, min_scene_len=5):
    """
    Detect keyframes using optical flow and save them as images.
    
    Parameters:
    - video_path: Path to the video file.
    - output_dir: Directory where keyframes will be saved.
    - threshold: Threshold for optical flow magnitude to detect significant motion.
    - min_scene_len: Minimum frame distance between consecutive keyframes.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_count = 0
    keyframe_count = 0
    keyframes = []

    first_frame_saved = False
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Save the last frame as a keyframe if it hasn't already been saved
            if last_frame is not None:
                last_frame_path = os.path.join(output_dir, f"frame_{frame_count - 1:05d}.jpg")
                cv2.imwrite(last_frame_path, last_frame)
                print(f"Saved last frame as keyframe: {last_frame_path}")
                keyframe_count += 1
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save the first frame as a keyframe
        if not first_frame_saved:
            first_frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(first_frame_path, frame)
            print(f"Saved first frame as keyframe: {first_frame_path}")
            keyframes.append(frame_count)
            keyframe_count += 1
            first_frame_saved = True

        # Calculate optical flow if there's a previous frame
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray_frame, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            # Calculate magnitude of optical flow
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(mag)

            # Detect significant motion
            if avg_magnitude > threshold:
                # Ensure minimum scene length between keyframes
                if len(keyframes) == 0 or (frame_count - keyframes[-1]) > min_scene_len:
                    keyframes.append(frame_count)
                    keyframe_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
                    cv2.imwrite(keyframe_path, frame)
                    keyframe_count += 1
                    print(f"Saved keyframe: {keyframe_path}")

        prev_frame = gray_frame
        last_frame = frame  # Store the current frame as a candidate for the last frame
        frame_count += 1

    cap.release()
    print(f"Detected and saved {keyframe_count} keyframes in '{output_dir}'.")


def extract_new_keyframes(video_path, existing_key_frame_folder, output_folder, histogram_diff_threshold=0.05, frame_sampling_interval=2):
    """
    Extract additional keyframes between existing ones based on histogram differences.
    
    Parameters:
    - video_path: Path to the video file.
    - existing_key_frame_folder: Directory containing existing keyframe images.
    - output_folder: Directory to save new keyframes.
    - histogram_diff_threshold: Threshold for histogram difference to detect significant changes.
    - frame_sampling_interval: Analyze every nth frame between keyframes.
    """
    os.makedirs(output_folder, exist_ok=True)
    existing_key_frame_folder= "/Users/shriyanshbhardwaj/Desktop/Mtech/py file /myenv/ keyframes_OF"
    # Function for natural sorting of filenames
    def natural_sort_key(filename):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

    # Load existing keyframes and extract frame indices
    key_frame_images = sorted(
        [f for f in os.listdir(existing_key_frame_folder) if f.endswith(('.jpg', '.png', '.jpeg'))],
        key=natural_sort_key
    )
    existing_key_frames = [int(os.path.splitext(img)[0].split('_')[-1]) for img in key_frame_images]

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Function to calculate histogram difference
    def calculate_histogram_difference(frame1, frame2):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return 1 - diff  # Lower correlation = more change

    # Process frames between existing keyframes
    new_key_frame_count = 0
    for i in range(len(existing_key_frames) - 1):
        start_frame = existing_key_frames[i]
        end_frame = existing_key_frames[i + 1]

        print(f"Analyzing frames between {start_frame} and {end_frame}...")

        # Seek to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = cap.read()
        if not ret:
            print(f"Error: Cannot read frame {start_frame}.")
            continue

        last_saved_frame_idx = start_frame
        for frame_idx in range(start_frame + 1, end_frame, frame_sampling_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate histogram difference
            diff = calculate_histogram_difference(prev_frame, frame)

            if diff > histogram_diff_threshold and (frame_idx - last_saved_frame_idx) > frame_sampling_interval:
                # Save the new keyframe
                new_key_frame_path = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(new_key_frame_path, frame)
                new_key_frame_count += 1
                print(f"New keyframe found at frame {frame_idx}.")

                # Update the last saved frame and prev_frame
                last_saved_frame_idx = frame_idx
                prev_frame = frame

    cap.release()
    print(f"New keyframe extraction completed. {new_key_frame_count} new keyframes saved in '{output_folder}'.")


def merge_files_to_local_folder(source_folder_1, source_folder_2, destination_folder_name):
    """
    Merge files from two source folders into a destination folder in the same directory as the Python script.

    Parameters:
    - source_folder_1: Path to the first source folder.
    - source_folder_2: Path to the second source folder.
    - destination_folder_name: Name of the destination folder to be created in the script's directory.
    """
    # Get the directory where the Python script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full path for the destination folder
    destination_folder = os.path.join(script_dir, destination_folder_name)
    
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Define the function to move files
    def move_files(source_folder, destination_folder):
        for filename in os.listdir(source_folder):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            
            # Move the file
            shutil.move(source_path, destination_path)

    # Merge files from both folders
    move_files(source_folder_1, destination_folder)
    move_files(source_folder_2, destination_folder)

    print(f"Files merged successfully into the folder: {destination_folder}")

def get_keyframes_from_folder(keyframe_folder):
    """Get the list of keyframe file paths from the folder."""
    keyframe_files = sorted([f for f in os.listdir(keyframe_folder) if f.endswith('.jpg') or f.endswith('.jpeg')])
    return keyframe_files

def get_video_frame_count(video_path):
    """Get the total number of frames in the video."""
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

def extract_frame_from_video(video_path, frame_number):
    """Extract a specific frame from the video."""
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    video.release()
    if ret:
        return frame
    else:
        return None

def add_middle_frame_between_keyframes(keyframe_folder, video_path, new_folder):
    """Add only one middle frame between keyframes if the difference in frames is greater than 10."""
    keyframe_files = get_keyframes_from_folder(keyframe_folder)
    total_frames = get_video_frame_count(video_path)
    
    print(f"Total frames in video: {total_frames}")

    # Extract the frame numbers of the existing keyframes
    keyframe_numbers = []
    for keyframe_file in keyframe_files:
        # Assuming the filename contains the frame number after the first underscore (e.g., "frame_00005.jpg")
        frame_number = int(keyframe_file.split('_')[1].split('.')[0])  # Extract frame number from filename
        keyframe_numbers.append(frame_number)

    print(f"Keyframe numbers: {keyframe_numbers}")

    # Iterate through the keyframes and add a middle frame in between if necessary
    for i in range(len(keyframe_numbers) - 1):
        current_keyframe = keyframe_numbers[i]
        next_keyframe = keyframe_numbers[i + 1]
        
        # Check if the difference in frame numbers is greater than 10
        if next_keyframe - current_keyframe > 10:
            # Find the middle frame number
            middle_frame_number = (current_keyframe + next_keyframe) // 2
            print(f"Adding middle frame between {current_keyframe} and {next_keyframe} (Middle frame: {middle_frame_number})...")

            # Extract the middle frame from the video
            middle_frame = extract_frame_from_video(video_path, middle_frame_number)
            if middle_frame is not None:
                # Save the new middle frame
                frame_filename = f"frame_{middle_frame_number:05d}.jpg"
                frame_filepath = os.path.join(new_folder, frame_filename)
                cv2.imwrite(frame_filepath, middle_frame)
                print(f"Added middle frame {frame_filename}")
            else:
                print(f"Failed to extract middle frame {middle_frame_number}")

    print("Added middle frames")



def compress_image_to_jpeg(image, quality=50):
    """
    Compress an image to JPEG format using OpenCV.
    Parameters:
        image (np.ndarray): The image to compress.
        quality (int): The quality of the JPEG compression (1-100).
    Returns:
        np.ndarray: Compressed image (in memory).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_image = cv2.imencode('.jpeg', image, encode_param)  # Compress to .jpeg
    return compressed_image

def compress_folder_to_jpeg(input_folder, output_folder, quality=50):
    """Compress all images in a folder to JPEG format using OpenCV's compression."""
    os.makedirs(output_folder, exist_ok=True)
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith((".jpg", ".png")):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            compressed_image = compress_image_to_jpeg(image, quality=quality)
            compressed_path = os.path.join(output_folder, filename.replace(".jpg", ".jpeg"))  # Save as .jpeg
            with open(compressed_path, 'wb') as f:
                f.write(compressed_image)  # Save the compressed .jpeg file
            print(f"Compressed and saved: {compressed_path}")

def create_video_from_images(folder_path, output_video_path, frame_rate=24):
    """
    Create a video from a folder of images.
    
    Parameters:
    - folder_path: Path to the folder containing image frames.
    - output_video_path: Path to save the output video.
    - frame_rate: Frame rate for the video (default is 24).
    """
    # Get the list of image files sorted by name
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if not image_files:
        print("No images found in the specified folder.")
        return

    # Get the size of the first image to determine video dimensions
    first_image_path = os.path.join(folder_path, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print("Error: Unable to read the first image.")
        return

    height, width, _ = frame.shape

    # Initialize the video writer with MKV codec
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # Codec for MKV format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each image frame to the video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Skipping {image_file}, unable to read.")
            continue
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video_path}")


def decompress_image_from_jpeg(compressed_image):
    """
    Decompress a JPEG image using OpenCV.
    Parameters:
        compressed_image (bytes): The compressed image (in memory).
    Returns:
        np.ndarray: Decompressed image.
    """
    # Convert bytes to a numpy array
    compressed_array = np.frombuffer(compressed_image, dtype=np.uint8)
    return cv2.imdecode(compressed_array, cv2.IMREAD_COLOR)

def decompress_folder_from_jpeg(input_folder, output_folder):
    """Decompress all images from JPEG format using OpenCV."""
    os.makedirs(output_folder, exist_ok=True)
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpeg"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'rb') as f:
                compressed_image = f.read()
            decompressed_image = decompress_image_from_jpeg(compressed_image)
            if decompressed_image is not None:
                decompressed_path = os.path.join(output_folder, filename.replace(".jpeg", ".jpg"))  # Save as .jpg
                cv2.imwrite(decompressed_path, decompressed_image)  # Save the decompressed .jpg
                print(f"Decompressed and saved: {decompressed_path}")
            else:
                print(f"Failed to decompress: {file_path}")

def proportional_interpolate_frames_half_with_keyframes(keyframes, frame_numbers, output_folder):
    """
    Generates half the number of interpolated frames proportional to the frame difference between keyframes,
    and saves both keyframes and generated frames in sequential order.

    Parameters:
    - keyframes (list of np.ndarray): List of keyframe images (as numpy arrays).
    - frame_numbers (list of int): Frame numbers corresponding to each keyframe.
    - output_folder (str): Path to save the interpolated frames and keyframes.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Start processing
    for i in range(len(keyframes) - 1):
        start_frame = keyframes[i]
        end_frame = keyframes[i + 1]
        start_number = frame_numbers[i]
        end_number = frame_numbers[i + 1]

        # Calculate the number of interpolated frames (half the difference)
        frame_diff = end_number - start_number
        num_frames = max(1, (frame_diff - 1) // 5)  # Ensure at least one frame is interpolated

        # Save the starting keyframe
        cv2.imwrite(os.path.join(output_folder, f"frame_{start_number:04d}.png"), start_frame)

        # Generate and save interpolated frames
        for j in range(1, num_frames + 1):
            alpha = j / (num_frames + 1)  # Interpolation factor
            interpolated_frame = ((1 - alpha) * start_frame + alpha * end_frame).astype(np.uint8)
            interpolated_frame_number = start_number + j * (frame_diff // (num_frames + 1))
            cv2.imwrite(os.path.join(output_folder, f"frame_{interpolated_frame_number:04d}.png"), interpolated_frame)

        # Save the ending keyframe
        cv2.imwrite(os.path.join(output_folder, f"frame_{end_number:04d}.png"), end_frame)


def load_keyframes(folder):
    """
    Loads keyframes from a folder and returns images and their frame numbers.

    Parameters:
    - folder (str): Path to the folder containing keyframe images.

    Returns:
    - list of np.ndarray: List of keyframe images.
    - list of int: List of frame numbers extracted from filenames.
    """
    keyframes = []
    frame_numbers = []

    # Load and sort keyframes by frame number
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            frame_number = int(os.path.splitext(filename)[0].split('_')[1])
            image = cv2.imread(os.path.join(folder, filename))
            keyframes.append(image)
            frame_numbers.append(frame_number)

    return keyframes, frame_numbers





if __name__ == "__main__":
    #change path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
    print(script_dir)
    video_path = '/Users/shriyanshbhardwaj/Desktop/Mtech/py_file/myenv/AlitaBattleAngel%20(online-video-cutter.mkv' 
    output_dir_OF = '/Users/shriyanshbhardwaj/Desktop/Mtech/py_file/myenv/keyframes_OF'  
    # # existing_key_frame_folder = "/Users/shriyanshbhardwaj/Desktop/Mtech/py file /myenv/ keyframes_OF"  
    # #1 
    keyframes_optical_flow(video_path, output_dir_OF, threshold=0.2, min_scene_len=5)
    # #2
    existing_key_frame_folder = os.path.join(script_dir, 'keyframes_OF')  # Folder with keyframes
    print(existing_key_frame_folder)

    output_folder_HD = os.path.join(script_dir, "output_folder_HD")
    os.makedirs(output_folder_HD, exist_ok=True)

    extract_new_keyframes(video_path, existing_key_frame_folder, output_folder_HD,
                          histogram_diff_threshold=0.05, frame_sampling_interval=2)
    
    merge_files_to_local_folder(existing_key_frame_folder, output_folder_HD, "Keyframes_OF_HD")
    keyframe_folder = os.path.join(script_dir, "keyframe_folder")

    os.makedirs(keyframe_folder, exist_ok=True)

    keyframe_final = os.path.join(script_dir, "keyframe_final")
    os.makedirs(keyframe_final, exist_ok=True)


    add_middle_frame_between_keyframes(keyframe_folder, video_path, keyframe_final)

    
    compressed_frames = os.path.join(script_dir, "compressed_frames")
    quality = 50  # JPEG quality (1-100)
    os.makedirs(keyframe_folder, exist_ok=True)
    compress_folder_to_jpeg(keyframe_final, compressed_frames, quality)

    output_video_path = os.path.join(script_dir, "video_compression.mkv")

    create_video_from_images(compressed_frames, output_video_path)

    decompressed_folder = os.path.join(script_dir, "decompressed_frames")
    os.makedirs(decompressed_folder, exist_ok=True)
    decompress_folder_from_jpeg(compressed_frames, decompressed_folder)


    # Load keyframes
    keyframes, frame_numbers = load_keyframes(decompressed_folder)

    output_interpolation = os.path.join(script_dir, "output_interpolation")
    os.makedirs(output_interpolation, exist_ok=True)
    # Interpolate and save frames
    proportional_interpolate_frames_half_with_keyframes(decompressed_folder, frame_numbers, output_interpolation)

    print(f"Frames (keyframes and interpolated) have been generated and saved to {output_interpolation}.")
    output_video_path_final = os.path.join(script_dir, "final_video.mkv")
    create_video_from_images(output_interpolation, output_video_path_final)


    




    

 





