import cv2 as cv
import os
import torch
import datetime
import tarfile
import numpy as np
import time
import subprocess
from typing import List, Optional
from diffusers import ControlNetModel, DiffusionPipeline, AutoPipelineForImage2Image
from latent_consistency_controlnet import LatentConsistencyModelPipeline_controlnet
from cog import BasePredictor, Input, Path
from PIL import Image

MODEL_CACHE_URL = "https://weights.replicate.delivery/default/fofr-lcm/model_cache.tar"
MODEL_CACHE = "model_cache"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def create_pipeline(
        self,
        pipeline_class,
        safety_checker: bool = True,
        controlnet: Optional[ControlNetModel] = None,
    ):
        kwargs = {
            "cache_dir": MODEL_CACHE,
            "local_files_only": True,
        }

        if not safety_checker:
            kwargs["safety_checker"] = None

        if controlnet:
            kwargs["controlnet"] = controlnet
            kwargs["scheduler"] = None

        pipe = pipeline_class.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", **kwargs)
        pipe.to(torch_device="cuda", torch_dtype=torch.float16)
        pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_CACHE_URL, MODEL_CACHE)

        self.txt2img_pipe = self.create_pipeline(DiffusionPipeline)
        self.txt2img_pipe_unsafe = self.create_pipeline(
            DiffusionPipeline, safety_checker=False
        )

        self.img2img_pipe = self.create_pipeline(AutoPipelineForImage2Image)
        self.img2img_pipe_unsafe = self.create_pipeline(
            AutoPipelineForImage2Image, safety_checker=False
        )

        controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",
            cache_dir="model_cache",
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.controlnet_pipe = self.create_pipeline(
            LatentConsistencyModelPipeline_controlnet, controlnet=controlnet_canny
        )
        self.controlnet_pipe_unsafe = self.create_pipeline(
            LatentConsistencyModelPipeline_controlnet,
            safety_checker=False,
            controlnet=controlnet_canny,
        )

        # warm the pipes
        self.txt2img_pipe(prompt="warmup")
        self.txt2img_pipe_unsafe(prompt="warmup")
        self.img2img_pipe(prompt="warmup", image=[Image.new("RGB", (768, 768))])
        self.img2img_pipe_unsafe(prompt="warmup", image=[Image.new("RGB", (768, 768))])
        self.controlnet_pipe(
            prompt="warmup",
            image=[Image.new("RGB", (768, 768))],
            control_image=[Image.new("RGB", (768, 768))],
        )
        self.controlnet_pipe_unsafe(
            prompt="warmup",
            image=[Image.new("RGB", (768, 768))],
            control_image=[Image.new("RGB", (768, 768))],
        )

    def control_image(self, image, canny_low_threshold, canny_high_threshold):
        image = np.array(image)
        canny = cv.Canny(image, canny_low_threshold, canny_high_threshold)
        return Image.fromarray(canny)

    def get_dimensions(self, image):
        original_width, original_height = image.size
        print(
            f"Original dimensions: Width: {original_width}, Height: {original_height}"
        )
        resized_width, resized_height = self.get_resized_dimensions(
            original_width, original_height
        )
        print(
            f"Dimensions to resize to: Width: {resized_width}, Height: {resized_height}"
        )
        return resized_width, resized_height

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

        if image and image.mode == "RGBA":
            image = image.convert("RGB")

        if control_image and control_image.mode == "RGBA":
            control_image = control_image.convert("RGB")

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

    @torch.inference_mode()
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
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        prediction_start = time.time()

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        print(f"Using seed: {seed}")

        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

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

        mode = "controlnet" if control_image else "img2img" if image else "txt2img"
        print(f"{mode} mode")

        pipe = getattr(
            self,
            f"{mode}_pipe" if not disable_safety_checker else f"{mode}_pipe_unsafe",
        )

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

        start = time.time()
        result = pipe(
            **common_args,
            **kwargs,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images
        print(f"Inference took: {time.time() - start:.2f}s")

        if archive_outputs:
            start = time.time()
            archive_start_time = datetime.datetime.now()
            print(f"Archiving images started at {archive_start_time}")

            tar_path = "/tmp/output_images.tar"
            with tarfile.open(tar_path, "w") as tar:
                for i, sample in enumerate(result):
                    output_path = f"/tmp/out-{i}.png"
                    sample.save(output_path)
                    tar.add(output_path, f"out-{i}.png")

            print(f"Archiving took: {time.time() - start:.2f}s")
            return Path(tar_path)

        # If not archiving, or there is an error in archiving, return the paths of individual images.
        output_paths = []
        for i, sample in enumerate(result):
            output_path = f"/tmp/out-{i}.jpg"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if canny_image:
            canny_image_path = "/tmp/canny-image.jpg"
            canny_image.save(canny_image_path)
            output_paths.append(Path(canny_image_path))

        print(f"Prediction took: {time.time() - prediction_start:.2f}s")
        return output_paths
