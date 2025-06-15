from __future__ import annotations
import math
import os
import random
import sys
import time
import einops
import gradio as gr
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler
sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config

# 禁用 Hugging Face 相关网络连接
os.environ["NO_PROXY"] = "huggingface.co,hf-mirror.com"

# GENERATATE_MODEL_PATH = r"./models/generate"
# EDIT_MODEL_PATH = r"./models/edit"
# GENERATED_IMG_PATH = r"./output_imgs/generated"
# EDITED_IMG_PATH = r"./output_imgs/edited" 

EDIT_RESOLUTION = 512
EDIT_VAE_CKPT_PATH = None 
EDIT_CONFIG_PATH = r"/root/autodl-tmp/ip2p_bs/configs/edit.yaml" 
GENERATATE_MODEL_PATH = r"/root/autodl-tmp/ip2p_bs/model/generate"
EDIT_MODEL_PATH = r"/root/autodl-tmp/ip2p_bs/model/edit"
GENERATED_IMG_PATH = r"/root/autodl-tmp/ip2p_bs/output_imgs/generated"
EDITED_IMG_PATH = r"/root/autodl-tmp/ip2p_bs/output_imgs/edited"

DETECT_MODEL_PATH = r".model/.."

title_text = "<h3><center>IMAGE GENERATION AND EDITING BASED ON TEXT-TO-IMAGE</center></h3>"

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

def get_model_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")
    files = os.listdir(path)
    model_files = [f for f in files if f.endswith(".ckpt") or f.endswith(".safetensors")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {path}.")
    
    return model_files

def load_editModel_from_config(config, edit_ckpt_path, vae_ckpt=None, verbose=False):
    print(f"Loading model from {edit_ckpt_path}") # TODO: 这里错了
    pl_sd = torch.load(edit_ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def load_generateModel(selected_model_name: str):
    model_path = os.path.join(GENERATATE_MODEL_PATH, selected_model_name)
    # 从单文件加载 Pipeline
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=False
    ).to("cuda")
    pipe.enable_vae_slicing()
    pipe.safety_checker = None
    return pipe

def load_editModel(edit_ckpt):

    # test
    print(f"Loading edit model from {EDIT_MODEL_PATH}/{edit_ckpt}")
    print(f"Loading edit config from {EDIT_CONFIG_PATH}")

    edit_config = OmegaConf.load(EDIT_CONFIG_PATH)
    edit_ckpt_path = os.path.join(EDIT_MODEL_PATH, edit_ckpt)
    edit_model = load_editModel_from_config(edit_config, edit_ckpt_path, EDIT_VAE_CKPT_PATH)
    edit_model.eval().cuda()

    model_wrap = K.external.CompVisDenoiser(edit_model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = edit_model.get_learned_conditioning([""])

    return edit_model, model_wrap, model_wrap_cfg, null_token

# EDIT_RESOLUTION = 512  # 编辑分辨率
def edit(
    input_image: Image.Image,
    instruction: str,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    edit_model,
    model_wrap,
    model_wrap_cfg,
    null_token
):
    if edit_model is None:  
        raise RuntimeError("请先从下拉框加载一个 edit 模型。")
    if input_image is None:
        raise ValueError("请先生成或上传一张图片。")
    if instruction is None or instruction.strip() == "":
        raise ValueError("请在编辑指令框中输入有效的指令。")

    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    width, height = input_image.size
    factor = EDIT_RESOLUTION / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if instruction == "":
        return [input_image, seed]

    with torch.no_grad(), autocast("cuda"), edit_model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [edit_model.get_learned_conditioning([instruction])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(edit_model.device)
        cond["c_concat"] = [edit_model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": text_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = edit_model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        return [seed, text_cfg_scale, image_cfg_scale, edited_image]

def generate(
    prompt: str,
    sampler: str = "DPM++ 2M",
    num_inference_steps: int = 50,
    randomize_scale: bool = True,
    guidance_scale: float = 7.5,
    randomize_seed: bool = True,
    seed: int = None,
    width: int = 512,
    height: int = 512,
    negative_prompt: str = None,
    pipe: StableDiffusionPipeline = None
):

    if pipe is None:
        raise RuntimeError("请先从生成模型下拉框加载一个模型")
    if prompt is None or prompt.strip() == "":
        raise ValueError("请在生成提示词框中输入有效的提示词。")

    guidance_scale = round(random.uniform(6.0, 16.0), ndigits=1) if randomize_scale else guidance_scale
    seed = random.randint(0, 100000) if randomize_seed else seed
    # 可选的 scheduler 映射
    schedulers = {
        "DPM++ 2M": DPMSolverMultistepScheduler,
        "DPM++ SDE": DPMSolverSinglestepScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "LMS": LMSDiscreteScheduler,
        "Euler": EulerDiscreteScheduler,
        # "DPM++ 2M Karras": DPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler"),
        "DPM2": KDPM2DiscreteScheduler,
        "DPM2 a": KDPM2AncestralDiscreteScheduler,
        "DDIM": DDIMScheduler,
        "DDPM": DDPMScheduler,
    }
    scheduler_cls = schedulers.get(sampler, DPMSolverMultistepScheduler)
    new_scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe.scheduler = new_scheduler

    generator = torch.Generator(device="cuda")
    generator = generator.manual_seed(seed)

    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    image = result.images[0]

    return [guidance_scale,seed,image]

def save_image(image: Image.Image, path: str):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    img_path = os.path.join(path, f"{timestamp}.png")
    filename = f"{path}/{timestamp}.png"
    image.save(filename)
    print(f"image saved in {filename}")

def reset_generateSetting():
    return ["DPM++ 2M", 100, 512, 512, "Fix CFG", 12, "Randomize Seed", 3397, None]

def reset_editSetting():
    return [50, "Randomize Seed", 1991, "Fix CFG", 7.5, 1.5, None]

def send_image_to_input_image(edited_image: Image.Image):
    if edited_image is None:
        return None
    return edited_image

def main():

    with gr.Blocks(css="footer {visibility: hidden}.big-button { height: 110px; font-size: 20px !important; }") as demo:
        
        state_edit_model = gr.State(None)
        state_model_wrap = gr.State(None)
        state_model_wrap_cfg = gr.State(None)
        state_null_token = gr.State(None)
        state_generate_pipe = gr.State(None)

        gr.Markdown(title_text)
       
        with gr.Row():
            generate_dropdown = gr.Dropdown(
                label="generation model list", 
                choices=get_model_list(GENERATATE_MODEL_PATH),
                elem_id="generate_model_dropdown"
            )
            edit_dropdown = gr.Dropdown(
                label="edition model list",
                choices=get_model_list(EDIT_MODEL_PATH),
                elem_id="edit_model_dropdown"
            )
        
        generate_dropdown.change(
            fn=load_generateModel,
            inputs=[generate_dropdown],
            outputs=[state_generate_pipe]
        )
        edit_dropdown.change(
            fn=load_editModel,
            inputs=edit_dropdown,
            outputs=[state_edit_model, state_model_wrap, state_model_wrap_cfg, state_null_token]
        )

        with gr.Row():
            with gr.Column():            
                generate_prompt = gr.Textbox(label="prompt", lines=3, placeholder="Prompt for generating image.\nUse ',' to separate keywords/tags.", elem_classes=["prompt"])
                generate_negative_prompt = gr.Textbox(label="negative prompt", lines=3, placeholder="Negative prompt for avoid these elements in the image.\nUse ',' to separate keywords/tags.", elem_classes=["prompt"])

            with gr.Column():
                with gr.Row():
                    instruction = gr.Textbox(lines=3, label="Edit Instruction", interactive=True)
                
                with gr.Row():
                        generate_button = gr.Button("Generate(text to image)",elem_classes=["big-button"])
                        edit_button = gr.Button("Edit(image to image)",elem_classes=["big-button"])
                
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    # 生成图片至此处 或 拖放上传其他图片至此处
                    input_image = gr.Image(label="Generated Image", type="pil", interactive=True, height=512, width=512)
                with gr.Row():           
                        save_generatedImage_button = gr.Button(scale=0.3,value="Save")                        
                        reset_generatedImage_button = gr.Button(scale=0.3,value="Reset")
            with gr.Column():
                with gr.Row():
                    # 修改图片后展示至此处
                    edited_image = gr.Image(label="Edited Image", type="pil", interactive=False, height=512, width=512)
                with gr.Row():
                    save_editedImage_button = gr.Button(value="Save")
                    # detect_editedImage_button = gr.Button("Detect")
                    reset_editedImage_button = gr.Button(value="Reset")
                    send_editedImage_button = gr.Button(value="Send")
                    

        with gr.Row():
            with gr.Accordion("Generation Settings", open=True):
                with gr.Row():
                    with gr.Column():
                        sampling_name = gr.Dropdown(label="Sampling method", choices=["DPM++ 2M", "DPM++ SDE", "Euler a", "LMS", "Euler", "DPM2", "DPM2 a", "DDIM" ,"DDPM"], value="DPM++ 2M", type="value", interactive=True)
                        sampling_steps = gr.Slider(label="Sampling steps", minimum=1, maximum=250, step=1, value=20)
                        
                    with gr.Column():
                        width = gr.Slider(label="Width", minimum=64, maximum=1024, step=64, value=512)
                        height = gr.Slider(label="Height", minimum=64, maximum=1024, step=64, value=512)

                    with gr.Row():
                        generate_randomize_cfg_scale = gr.Radio(
                            ["Fix CFG", "Randomize CFG"],
                            value="Fix CFG",
                            type="index",
                            show_label=False,
                            interactive=True,
                        )
                        generate_cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=20.0, value=12,interactive=True,step=0.1)
                        # generate_cfg_scale = gr.Number(value=12, label="CFG Scale", interactive=True)
                        generate_randomize_seed = gr.Radio(
                            ["Fix Seed", "Randomize Seed"],
                            value="Randomize Seed",
                            type="index",
                            show_label=False,
                            interactive=True,
                        )
                        generate_seed = gr.Number(value=1099, precision=0, label="Seed", interactive=True)

        
            # insturction-pix2pix 的设置参数
            with gr.Accordion("Edition Settings", open=False):
                with gr.Column():
                    with gr.Row():
                        edit_steps = gr.Number(value=100, precision=0, label="Steps", interactive=True)
                        randomize_cfg = gr.Radio(
                            ["Fix CFG", "Randomize CFG"],
                            value="Fix CFG",
                            type="index",
                            show_label=False,
                            interactive=True,
                        )
                        text_cfg_scale = gr.Number(value=7.5, label="Text CFG", interactive=True)
                        image_cfg_scale = gr.Number(value=1.5, label="Image CFG", interactive=True)
                    with gr.Row():
                        edit_randomize_seed = gr.Radio(
                            ["Fix Seed", "Randomize Seed"],
                            value="Randomize Seed",
                            type="index",
                            show_label=False,
                            interactive=True,
                        )
                        edit_seed = gr.Number(value=3069, precision=0, label="Seed", interactive=True)

        generate_button.click(
            fn=generate,
            # fn=generate_test,
            inputs=[
                generate_prompt,
                sampling_name,
                sampling_steps,
                generate_randomize_cfg_scale,
                generate_cfg_scale,
                generate_randomize_seed,
                generate_seed,
                width,
                height,
                generate_negative_prompt,
                state_generate_pipe
            ],
            outputs=[generate_cfg_scale, generate_seed, input_image],
        )

        edit_button.click(
            fn=edit,
            inputs=[
                input_image,
                instruction,
                edit_steps,
                edit_randomize_seed,
                edit_seed,
                randomize_cfg,
                text_cfg_scale,
                image_cfg_scale,
                state_edit_model, state_model_wrap, state_model_wrap_cfg, state_null_token
            ],
            outputs=[edit_seed, text_cfg_scale, image_cfg_scale, edited_image],
        )

        save_generatedImage_button.click(
            fn=lambda img: save_image(img, GENERATED_IMG_PATH),
            inputs=[input_image],
            outputs=[]
        )

        save_editedImage_button.click(
            fn=lambda img: save_image(img, EDITED_IMG_PATH),
            inputs=[edited_image],
            outputs=[]
        )

        reset_generatedImage_button.click(
            fn=reset_generateSetting,
            inputs=[],
            outputs=[sampling_name, sampling_steps, width, height, generate_randomize_cfg_scale, generate_cfg_scale, generate_randomize_seed, generate_seed, input_image],
        )

        reset_editedImage_button.click(
            fn=reset_editSetting,
            inputs=[],
            outputs=[edit_steps, edit_randomize_seed, edit_seed, randomize_cfg, text_cfg_scale, image_cfg_scale, edited_image],
        )

        send_editedImage_button.click(
            fn=send_image_to_input_image,
            inputs=[edited_image],
            outputs=[input_image]
        )

    demo.queue(max_size=1)
    # demo.queue(concurrency_count=1)
    demo.launch(share=True)


if __name__ == "__main__":
    main()



# 评估标准
# 对模型生成结果进行了定量评估与用户体验测试，

