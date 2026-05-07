import sys
import imageio

def convert_webp_to_mp4(input_path, output_path):
    print(f"Reading {input_path}...")
    try:
        # Read the WebP frames
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 10) # default to 10 fps if not found
        
        print(f"Writing to {output_path} at {fps} FPS...")
        # Write to MP4 using ffmpeg plugin from imageio
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=None)
        
        count = 0
        for frame in reader:
            writer.append_data(frame)
            count += 1
            if count % 50 == 0:
                print(f"Processed {count} frames...")
                
        writer.close()
        print(f"Success! Saved MP4 with {count} frames to {output_path}")
    except Exception as e:
        print(f"Error converting video: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_video.py <input.webp> <output.mp4>")
        sys.exit(1)
        
    convert_webp_to_mp4(sys.argv[1], sys.argv[2])
