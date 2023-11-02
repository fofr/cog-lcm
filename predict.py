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
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
            cache_dir="model_cache",
            safety_checker=None,
            local_files_only=True
        )

        self.txt2img_pipe.to(torch_device="cuda", torch_dtype=torch.float16)

        self.img2img_pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline=".",
            cache_dir="model_cache",
            safety_checker=None,
            local_files_only=True
        )

        self.img2img_pipe.to(torch_device="cuda", torch_dtype=torch.float16)

    def images_to_video(self, image_folder_path, output_video_path, fps):
        # Forming the ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),  # Set the framerate for the input files
            '-pattern_type', 'glob',  # Enable pattern matching for filenames
            '-i', f'{image_folder_path}/out-*.png',  # Input files pattern
            '-c:v', 'libx264',  # Set the codec for video
            '-pix_fmt', 'yuv420p',  # Set the pixel format
            '-crf', '17',  # Set the constant rate factor for quality
            output_video_path  # Output file
        ]

        # Run the ffmpeg command
        subprocess.run(cmd)

    def zoom_image(self, image: Image.Image, zoom_percentage: float) -> Image.Image:
        """Zooms into the image by a given percentage."""
        width, height = image.size
        new_width = width * (1 + zoom_percentage)
        new_height = height * (1 + zoom_percentage)

        # Resize the image to the new dimensions
        zoomed_image = image.resize((int(new_width), int(new_height)))

        # Crop the image to the original dimensions, focusing on the center
        left = (zoomed_image.width - width) / 2
        top = (zoomed_image.height - height) / 2
        right = (zoomed_image.width + width) / 2
        bottom = (zoomed_image.height + height) / 2

        return zoomed_image.crop((left, top, right, bottom))

    def tar_frames(self, frame_paths, tar_path):
        with tarfile.open(tar_path, "w:gz") as tar:
            for frame in frame_paths:
                tar.add(frame)

    def predict(
        self,
        start_prompt: str = Input(
            description="Prompt to start with, if not using an image",
            default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        ),
        end_prompt: str = Input(
            description="Prompt to animate towards",
            default="Self-portrait watercolour, a beautiful cyborg with purple hair, 8k",
        ),
        image: Path = Input(
            description="Starting image if not using a prompt",
            default=None,
        ),
        width: int = Input(
            description="Width of output. Lower if out of memory",
            default=512,
        ),
        height: int = Input(
            description="Height of output. Lower if out of memory",
            default=512,
        ),
        iterations: int = Input(
            description="Number of times to repeat the img2img pipeline",
            default=12,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.2,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommend 1 to 8 steps.",
            ge=1,
            le=50,
            default=8,
        ),
        zoom_increment: int = Input(
            description="Zoom increment percentage for each frame",
            ge=0,
            le=4,
            default=0,

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
        tmp_frames = glob.glob("/tmp/out-*.png")
        for frame in tmp_frames:
            os.remove(frame)

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        common_args = {
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "lcm_origin_steps": 50,
            "output_type": "pil"
        }

        img2img_args = {
            "num_inference_steps": num_inference_steps,
            "prompt": end_prompt,
            "strength": prompt_strength
        }

        if image:
            print("img2img mode")
            img2img_args["image"] = Image.open(image)
        else:
            print("txt2img mode")
            txt2img_args = {
                "prompt": start_prompt,
                "num_inference_steps": 8 # Always want a good starting image
            }
            result = self.txt2img_pipe(**common_args, **txt2img_args).images
            img2img_args["image"] = result[0]

        last_image_path = None
        frame_paths = []

        # Iteratively applying img2img transformations
        for iteration in range(iterations):
            if last_image_path:
                print(f"img2img iteration {iteration}")
                img2img_args["image"] = Image.open(last_image_path)

                zoom_increment_mapping = {4: 0.1, 3: 0.05, 2: 0.025, 1: 0.00125}
                if 1 <= zoom_increment <= 4:
                    zoom_factor = zoom_increment_mapping[zoom_increment]
                    img2img_args["image"] = self.zoom_image(img2img_args["image"], zoom_factor)

            # Execute the model pipeline here
            result = self.img2img_pipe(**common_args, **img2img_args).images

            # Save the resulting image for the next iteration
            last_image_path = f"/tmp/out-{iteration:06d}.png"
            result[0].save(last_image_path)
            frame_paths.append(last_image_path)

        # Creating an mp4 video from the images
        video_path = "/tmp/output_video.mp4"
        self.images_to_video("/tmp", video_path, 12)

        # Tar and return all the frames if return_frames is True
        if return_frames:
            print(f"Tarring and returning all frames")
            tar_path = "/tmp/frames.tar.gz"
            self.tar_frames(frame_paths, tar_path)
            return [Path(video_path), Path(tar_path)]

        return [Path(video_path)]
