import cv2 as cv
import os
import torch
import datetime
import tarfile
import numpy as np
from typing import List
from diffusers import ControlNetModel, DiffusionPipeline, AutoPipelineForImage2Image
from latent_consistency_controlnet import LatentConsistencyModelPipeline_controlnet
from cog import BasePredictor, Input, Path
from PIL import Image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        torch_device = "cuda"
        torch_dtype = torch.float16

        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            cache_dir="model_cache",
            local_files_only=True,
        )

        self.txt2img_pipe.to(torch_device=torch_device, torch_dtype=torch_dtype)

        self.img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            cache_dir="model_cache",
            local_files_only=True,
        )

        self.img2img_pipe.to(torch_device=torch_device, torch_dtype=torch_dtype)

        controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            cache_dir="model_cache",
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(torch_device)

        self.controlnet_pipe = (
            LatentConsistencyModelPipeline_controlnet.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                cache_dir="model_cache",
                local_files_only=True,
                safety_checker=None,
                controlnet=controlnet_canny,
                scheduler=None,
            )
        )

        self.controlnet_pipe.to(torch_device=torch_device, torch_dtype=torch_dtype)

    def control_image(self, image, canny_low_threshold, canny_high_threshold):
        image = np.array(image)
        canny = cv.Canny(image, canny_low_threshold, canny_high_threshold)
        return Image.fromarray(canny)

    def get_allowed_dimensions(self, base=512, max_dim=1024):
        """
        Function to generate allowed dimensions optimized around a base up to a max
        """
        allowed_dimensions = []
        for i in range(base, max_dim + 1, 64):
            for j in range(base, max_dim + 1, 64):
                allowed_dimensions.append((i, j))
        return allowed_dimensions

    def get_resized_dimensions(self, width, height):
        """
        Function adapted from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        allowed_dimensions = self.get_allowed_dimensions()
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        # and are closest to the optimum dimension of 768
        optimum_dimension = 768
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
            + abs(dim[0] - optimum_dimension),
        )
        return closest_dimensions

    def resize_images(self, images, width, height):
        return [
            img.resize((width, height)) if img is not None else None for img in images
        ]

    def open_image(self, image_path):
        return Image.open(str(image_path)) if image_path is not None else None

    def apply_sizing_strategy(
        self, sizing_strategy, width, height, control_image=None, image=None
    ):
        image = self.open_image(image)
        control_image = self.open_image(control_image)

        if sizing_strategy == "input_image":
            print("Resizing based on input image")
            width, height = self.get_dimensions(image)
        elif sizing_strategy == "control_image":
            print("Resizing based on control image")
            width, height = self.get_dimensions(control_image)
        else:
            print("Using given dimensions")

        image, control_image = self.resize_images([image, control_image], width, height)
        return width, height, control_image, image

    def predict(
        self,
        prompt: str = Input(
            description="For multiple prompts, enter each on a new line.",
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
        sizing_strategy: str = Input(
            description="Decide how to resize images – use width/height, resize based on input image or control image",
            choices=["width/height", "input_image", "control_image"],
            default="width/height",
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
            description="Number of images per prompt",
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
        ),
        control_image: Path = Input(
            description="Image for controlnet conditioning",
            default=None,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Controlnet conditioning scale",
            ge=0.1,
            le=4.0,
            default=2.0,
        ),
        control_guidance_start: float = Input(
            description="Controlnet start",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        control_guidance_end: float = Input(
            description="Controlnet end",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        canny_low_threshold: float = Input(
            description="Canny low threshold",
            ge=1,
            le=255,
            default=100,
        ),
        canny_high_threshold: float = Input(
            description="Canny high threshold",
            ge=1,
            le=255,
            default=200,
        ),
        archive_outputs: bool = Input(
            description="Option to archive the output images",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        prompt = prompt.strip().splitlines()
        if len(prompt) == 1:
            print("Found 1 prompt:")
        else:
            print(f"Found {len(prompt)} prompts:")
        for p in prompt:
            print(f"- {p}")

        if len(prompt) * num_images == 1:
            print("Making 1 image")
        else:
            print(f"Making {len(prompt) * num_images} images")

        if image or control_image:
            (
                width,
                height,
                control_image,
                image,
            ) = self.apply_sizing_strategy(
                sizing_strategy, width, height, control_image, image
            )

        kwargs = {}
        canny_image = None

        if image:
            kwargs["image"] = image
            kwargs["strength"] = prompt_strength

        if control_image:
            canny_image = self.control_image(
                control_image, canny_low_threshold, canny_high_threshold
            )
            kwargs["control_guidance_start"]: control_guidance_start
            kwargs["control_guidance_end"]: control_guidance_end
            kwargs["controlnet_conditioning_scale"]: controlnet_conditioning_scale

            # TODO: This is a hack to get controlnet working without an image input
            # The current pipeline doesn't seem to support not having an image, so
            # we pass one in but set strength to 1 to ignore it
            if not image:
                kwargs["image"] = Image.new("RGB", (width, height), (128, 128, 128))
                kwargs["strength"] = 1.0

            kwargs["control_image"] = canny_image

        if control_image:
            print("controlnet mode")
            pipe = self.controlnet_pipe
        elif image:
            print("img2img mode")
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
            "output_type": "pil",
        }
        result = pipe(**common_args, **kwargs).images

        if archive_outputs:
            archive_start_time = datetime.datetime.now()
            print(f"Archiving images started at {archive_start_time}")

            tar_path = "/tmp/output_images.tar"
            with tarfile.open(tar_path, "w") as tar:
                for i, sample in enumerate(result):
                    output_path = f"/tmp/out-{i}.png"
                    sample.save(output_path)
                    tar.add(output_path, f"out-{i}.png")

            return Path(tar_path)

        # If not archiving, or there is an error in archiving, return the paths of individual images.
        output_paths = []
        for i, sample in enumerate(result):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if canny_image:
            canny_image_path = "/tmp/canny-image.png"
            canny_image.save(canny_image_path)
            output_paths.append(Path(canny_image_path))

        return output_paths
