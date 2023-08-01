# from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from diffusers.utils import load_image
# import torch
# from PIL import Image
# base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# controlnet_path = "controlnet_sdxl_lineart/checkpoint-500"

# controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32, subfolder='controlnet')
# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#     base_model_path, controlnet=controlnet, torch_dtype=torch.float32
# ).to('cuda:3')

# # speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# # remove following line if xformers is not installed or when using Torch 2.0.
# # pipe.enable_xformers_memory_efficient_attention()
# # memory optimization.
# # pipe.enable_model_cpu_offload()

# control_image = load_image("https://datasets-server.huggingface.co/assets/zbulrush/lineart/--/default/train/2/conditioning_image/image.jpg")
# prompt = "a close photo up of a car with a stylish metal wheel"
# control_image.save("original.png")
# # generate image
# generator = torch.manual_seed(1000)
# image = pipe(
#     prompt, num_inference_steps=30, generator=generator, image=control_image
# ).images[0]
# image.save("./output.png")