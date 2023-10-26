import os
import torch
import cv2
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
            local_files_only=True
        )

        self.txt2img_pipe.to(torch_device="cuda", torch_dtype=torch.float16)

        self.img2img_pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline=".",
            cache_dir="model_cache",
            local_files_only=True
        )

        self.img2img_pipe.to(torch_device="cuda", torch_dtype=torch.float16)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        ),
        width: int = Input(
            description="Width of output image. Lower if out of memory",
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Lower if out of memory",
            default=768,
        ),
        iterations: int = Input(
            description="Number of times to repeat the img2img pipeline",
            default=1,
        ),
        image: Path = Input(
            description="Input image for img2img",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        num_images: int = Input(
            description="Number of images to output",
            ge=1,
            le=50,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommend 1 to 8 steps.",
            ge=1,
            le=50,
            default=8,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=8.0
        ),
        lcm_origin_steps: int = Input(
            ge=1,
            default=50,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        )
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        kwargs = {}
        if image:
            print("img2img mode")
            input_image = Image.open(image)
            kwargs["image"] = input_image
            kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            pipe = self.txt2img_pipe

        common_args = {
            "width": width,
            "height": height,
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "num_inference_steps": num_inference_steps,
            "lcm_origin_steps": lcm_origin_steps,
            "output_type": "pil"
        }

        # Initialization of the last_image_path variable
        last_image_path = None

        # Iteratively applying img2img transformations
        for iteration in range(iterations):
            if last_image_path:
                print(f"img2img iteration {iteration}")
                input_image = Image.open(last_image_path)
                kwargs["image"] = input_image

            # Execute the model pipeline here
            result = pipe(**common_args, **kwargs).images

            # Save the resulting image for the next iteration
            last_image_path = f"/tmp/out-{iteration}.png"
            result[0].save(last_image_path)

        # Creating an mp4 video from the images
        video_path = "/tmp/output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 12.0, (width, height))

        # Adding images to the video
        for iteration in range(iterations):
            img_path = f"/tmp/out-{iteration}.png"
            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not load image at path: {img_path}")
                continue

            # Converting color space from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resizing the image to match the video's dimensions
            img_resized = cv2.resize(img_rgb, (width, height))
            out.write(img_resized)

        out.release()  # Finalize the video file

        if not os.path.exists(video_path):
            print(f"Video could not be saved at path: {video_path}")
            return None

        return Path(video_path)
