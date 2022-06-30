import tempfile
import numpy as np
import torch
from PIL import Image
from typing import List
from cog import BasePredictor, Path, Input, BaseModel

from inference_VQ_Diffusion import VQ_Diffusion

with open("imagenet_class.txt") as infile:
    CATEGORY = [line.rstrip() for line in infile]


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):
    def setup(self):

        self.VQ_Diffusion_model_condition = VQ_Diffusion_Cog(
            config="configs/ithq.yaml",
            path="OUTPUT/pretrained_model/ithq_learnable.pth",
        )

        self.VQ_Diffusion_model_condition_coco = VQ_Diffusion_Cog(
            config="OUTPUT/pretrained_model/config_text.yaml",
            path="OUTPUT/pretrained_model/coco_learnable.pth",
        )

        self.VQ_Diffusion_model_class = VQ_Diffusion_Cog(
            config="OUTPUT/pretrained_model/config_imagenet.yaml",
            path="OUTPUT/pretrained_model/imagenet_pretrained.pth",
        )

    def predict(
        self,
        generation_type: str = Input(
            choices=["in-the-wild text", "MSCOCO datasets", "ImageNet class label"],
            default="in-the-wild text",
            description="Choose generating images from in-the-wild text, MSCOCO datasets, or ImageNet class label.",
        ),
        prompt: str = Input(
            default="",
            description="Prompt for generating image. Valid when generation_type is set to in-the-wild text and MSCOCO datasets.",
        ),
        image_class: str = Input(
            choices=CATEGORY + ["None"],
            default="None",
            description="Choose the ImageNet label. Valid when generation_type is set to ImageNet class label.",
        ),
        num_images: int = Input(
            default=4,
            ge=1,
            le=8,
            description="Number of generated images.",
        ),
        truncation_rate: float = Input(
            default=0.86,
            ge=0.0,
            le=1.0,
            description="Sample with truncation.",
        ),
        guidance_scale: float = Input(
            default=1.0,
            description="Improved VQ-Diffusion with learnable classifier-free sampling.",
        ),
    ) -> List[ModelOutput]:
        if generation_type == "ImageNet class label":
            assert not image_class == "None", "Please specify a class label."
        else:
            assert len(prompt) > 0, "Please provide a prompt."

        if generation_type == "ImageNet class label":
            VQ_Diffusion_model = self.VQ_Diffusion_model_class
            image_class = int(image_class.split(") ")[0])
            images = VQ_Diffusion_model.generate_sample_with_class(
                image_class,
                truncation_rate=truncation_rate,
                batch_size=num_images,
                guidance_scale=guidance_scale,
            )
        elif generation_type == "in-the-wild text":
            VQ_Diffusion_model = self.VQ_Diffusion_model_condition
            images = VQ_Diffusion_model.generate_sample_with_condition(
                prompt,
                truncation_rate=truncation_rate,
                batch_size=num_images,
                guidance_scale=guidance_scale,
            )
        else:
            VQ_Diffusion_model = self.VQ_Diffusion_model_condition_coco
            images = VQ_Diffusion_model.generate_sample_with_condition(
                prompt,
                truncation_rate=truncation_rate,
                batch_size=num_images,
                guidance_scale=guidance_scale,
            )
        output = []

        for i, img in enumerate(images):
            output_path = Path(tempfile.mkdtemp()) / f"output_{i}.png"
            img.save(str(output_path))
            output.append(ModelOutput(image=output_path))

        return output


class VQ_Diffusion_Cog(VQ_Diffusion):
    def generate_sample_with_class(
        self, text, truncation_rate, batch_size, infer_speed=False, guidance_scale=1.0
    ):

        self.model.guidance_scale = guidance_scale

        data_i = {"label": [text], "image": None}
        condition = text

        str_cond = str(condition)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top" + str(truncation_rate) + "r",
            )  # B x C x H x W

        # save results
        content = model_out["content"]
        content = content.permute(0, 2, 3, 1).to("cpu").numpy().astype(np.uint8)
        images = []
        for b in range(content.shape[0]):
            im = Image.fromarray(content[b])
            images.append(im)
        return images

    def generate_sample_with_condition(
        self,
        text,
        truncation_rate,
        batch_size,
        infer_speed=False,
        guidance_scale=1.0,
        prior_rule=0,
        prior_weight=0,
        learnable_cf=True,
    ):

        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = learnable_cf
        self.model.transformer.prior_rule = prior_rule
        self.model.transformer.prior_weight = prior_weight

        data_i = {"text": [text], "image": None}
        condition = text

        str_cond = str(condition)

        if infer_speed != False:
            add_string = "r,time" + str(infer_speed)
        else:
            add_string = "r"
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top" + str(truncation_rate) + add_string,
            )  # B x C x H x W

        # save results
        content = model_out["content"]
        content = content.permute(0, 2, 3, 1).to("cpu").numpy().astype(np.uint8)
        images = []
        for b in range(content.shape[0]):
            im = Image.fromarray(content[b])
            images.append(im)
        return images
