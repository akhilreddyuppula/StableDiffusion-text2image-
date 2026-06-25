from diffusers import StableDiffusionXLPipeline
import torch


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    cache_dir="./models"
)

generator = torch.Generator("cuda").manual_seed(67)

pipe = pipe.to("cuda")

prompt = "dog dressed up as a wizard"
image = pipe(
    prompt=prompt,
    num_inference_steps=40,
    height=1024,
    width=1024,
    guidance_scale=7.5,
    generator=generator
).images[0]

image.save("generatedimage-2.png")

