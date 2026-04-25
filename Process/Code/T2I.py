from diffusers import StableDiffusion3Pipeline
import torch

model_path = "stable-diffusion-3-medium-diffusers"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

image = pipe(
    "a cat wearing a sunglasses and a dog wearing a hat",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]

image.save("sd3_t2i.png")