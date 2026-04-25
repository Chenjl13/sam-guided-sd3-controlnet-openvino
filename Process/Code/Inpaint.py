import torch
from diffusers import StableDiffusion3InpaintPipeline
from diffusers.utils import load_image

model_path = "stable-diffusion-3-medium-diffusers"

pipe = StableDiffusion3InpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "A Tiger and a rabbit"
#negative_prompt = "blurry, distorted, low quality"

image = load_image("sd3_t2i.png").convert("RGB")  ## Original Image
mask_image = load_image("mask_1.png").convert("L")  ## Mask

image = image.resize((1024, 1024))
mask_image = mask_image.resize((1024, 1024))

result = pipe(
    prompt=prompt,
    #negative_prompt=negative_prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]

result.save("sd3_inpaint.png")
print("saved to sd3_inpaint_result.png")