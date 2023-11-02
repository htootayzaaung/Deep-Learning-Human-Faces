import os
import subprocess

def compress_video(input_path):
    # Get the base name of the original video (without extension)
    base_name = os.path.basename(input_path).split('.')[0]
    
    # Define the output path
    output_directory = "Videos/Compressed/"
    output_path = os.path.join(output_directory, f"{base_name}.mp4")
    
    # Make sure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # ffmpeg command to compress the video
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', 'scale=iw/2:ih/2',
        '-b:v', '1000k',
        '-c:v', 'libx264',
        '-preset', 'fast',
        output_path
    ]
    
    # Execute the command
    subprocess.run(cmd)

    print(f"Video compressed and saved to {output_path}")


# Example usage
input_video_path = "Videos/Bella_Ciao.mp4"
compress_video(input_video_path)
