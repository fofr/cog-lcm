import os
import torch
import datetime
import tarfile
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
        ),
        archive_outputs: bool = Input(
            description="Option to archive the output images",
            default=False,
        )
    ) -> list[Path]:
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
        result = pipe(**common_args, **kwargs).images

        if archive_outputs:
            archive_start_time = datetime.datetime.now()
            print(f"Archiving images started at {archive_start_time}")

            tar_path = "/tmp/output_images.tar"
            with tarfile.open(tar_path, 'w') as tar:
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

        return output_paths
