import os
import torch
import subprocess
import glob
import tarfile
from typing import List
from diffusers import DiffusionPipeline
from pipeline import LatentConsistencyModelImg2ImgPipeline
from cog import BasePredictor, Input, Path
from PIL import Image

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.img2img_pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline=".",
            cache_dir="model_cache",
            safety_checker=None,
            local_files_only=True
        )

        self.img2img_pipe.to(torch_device="cuda", torch_dtype=torch.float16)

    def extract_frames(self, video, fps, extract_all_frames):
        os.makedirs("/tmp", exist_ok=True)

        if not extract_all_frames:
            command = f'ffmpeg -i "{video}" -vf fps={fps} /tmp/out%03d.png'
        else:
            command = f'ffmpeg -i "{video}" /tmp/out%03d.png'

        subprocess.run(command, shell=True, check=True)
        frame_files = sorted(os.listdir("/tmp"))
        frame_files = [file for file in frame_files if file.endswith('.png') and 'out' in file]

        print(f"Extracted {len(frame_files)} frames from video")
        return [f"/tmp/{frame_file}" for frame_file in frame_files]

    def width_height(self, frame_paths):
        img = Image.open(frame_paths[0])
        width, height = img.size
        return width, height

    def resize_frames(self, frame_paths, max_width):
        for frame_path in frame_paths:
            img = Image.open(frame_path)
            width, height = img.size
            if width > max_width:
                height = int(height * max_width / width)
                width = max_width
                img = img.resize((width, height))
                img.save(frame_path)

    def images_to_video(self, image_folder_path, output_video_path, fps):
        # Forming the ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),  # Set the framerate for the input files
            '-pattern_type', 'glob',  # Enable pattern matching for filenames
            '-i', f'{image_folder_path}/out*.png',  # Input files pattern
            '-c:v', 'libx264',  # Set the codec for video
            '-pix_fmt', 'yuv420p',  # Set the pixel format
            '-crf', '17',  # Set the constant rate factor for quality
            output_video_path  # Output file
        ]

        # Run the ffmpeg command
        subprocess.run(cmd)

    def tar_frames(self, frame_paths, tar_path):
        with tarfile.open(tar_path, "w:gz") as tar:
            for frame in frame_paths:
                tar.add(frame)

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for video2video",
            default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        ),
        video: Path = Input(
            description="Video to split into frames"
        ),
        fps: int = Input(
            description="Number of images per second of video, when not exporting all frames",
            default=8,
            ge=1
        ),
        extract_all_frames: bool = Input(
            description="Get every frame of the video. Ignores fps. Slow for large videos.",
            default=False
        ),
        max_width: int = Input(
            description="Maximum width of the video. Maintains aspect ratio.",
            default=512,
            ge=1
        ),
        prompt_strength: float = Input(
            description="1.0 corresponds to full destruction of information in video frame",
            ge=0.0,
            le=1.0,
            default=0.2,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps per frame. Recommend 1 to 8 steps.",
            ge=1,
            le=50,
            default=4,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=8.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        return_frames: bool = Input(
            description="Return a tar file with all the frames alongside the video", default=False
        )
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Removing all temporary frames
        tmp_frames = glob.glob("/tmp/out*.png")
        for frame in tmp_frames:
            os.remove(frame)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        # Extract frames from video
        print(f"Extracting frames from video: {video}")
        frame_paths = self.extract_frames(video, fps, extract_all_frames)

        # Resize frames
        print(f"Resizing frames to max width: {max_width}")
        self.resize_frames(frame_paths, max_width)

        width, height = self.width_height(frame_paths)

        img2img_args = {
            "num_inference_steps": num_inference_steps,
            "prompt": prompt,
            "strength": prompt_strength,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "lcm_origin_steps": 50,
            "output_type": "pil"
        }

        # Run img2img pipeline on each frame
        print(f"Running img2img pipeline on each frame")
        for frame_path in frame_paths:
            img2img_args["image"] = Image.open(frame_path)
            result = self.img2img_pipe(**img2img_args).images
            print(f"Saving frame: {frame_path}")
            result[0].save(frame_path)

        # Create a new video from the frames
        print(f"Creating video from frames")
        video_path = "/tmp/output_video.mp4"
        self.images_to_video("/tmp", video_path, fps)

        # Tar and return all the frames if return_frames is True
        if return_frames:
            print(f"Tarring and returning all frames")
            tar_path = "/tmp/frames.tar.gz"
            self.tar_frames(frame_paths, tar_path)
            return [Path(video_path), Path(tar_path)]

        return [Path(video_path)]
