import torch
import joblib
from PIL import Image
import numpy as np
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from utils import zero_out, apply_flux_guidance
from apply_clip import run_clip
from apply_style import load_style_model, apply_stylemodel, STYLE_MODEL_PATH, CLIPOutputWrapper

def prepare_embeddings_for_diffusers(positive_conds, negative_conds):
    uncond_prompt_embeds = negative_conds[0][0]
    cond_prompt_embeds = positive_conds[0][0]
    prompt_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds], dim=0)

    pool_key = 'pooled_output'
    if pool_key not in positive_conds[0][1] or pool_key not in negative_conds[0][1]:
        raise ValueError(f"Could not find key '{pool_key}' in the conditioning dictionary.")

    uncond_pooled_embeds = negative_conds[0][1][pool_key]
    cond_pooled_embeds = positive_conds[0][1][pool_key]
    pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, cond_pooled_embeds], dim=0)
    
    return prompt_embeds, pooled_prompt_embeds

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    print(f"Using device: {device} with dtype: {dtype}")

    print("\n--- Step 1: Loading and preparing cached text embeddings ---")
    conditioning_cache_path = "/teamspace/studios/this_studio/.porting/models/conditioning/text_cache.conditioning"
    conditioning_cpu = joblib.load(conditioning_cache_path)
    
    conditioning = []
    for tensor_part, dict_part in conditioning_cpu:
        tensor_part = tensor_part.to(device)
        new_dict_part = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict_part.items()}
        conditioning.append([tensor_part, new_dict_part])

    clip_vis_tensor = run_clip().to(device)
    clip_vis_output = CLIPOutputWrapper(clip_vis_tensor)
    style_model = load_style_model(STYLE_MODEL_PATH)
    style_model.model = style_model.model.to(device)
    styled_conds, = apply_stylemodel(conditioning, style_model, clip_vis_output, 1, "multiply")

    negative_conds, = zero_out(styled_conds)
    positive_conds, = apply_flux_guidance(styled_conds, 40.3)
    
    prompt_embeds, pooled_prompt_embeds = prepare_embeddings_for_diffusers(
        positive_conds, negative_conds
    )
    print("Cached embeddings prepared successfully.")
    
    print("\n--- Step 2: Fusing image and text guidance with FluxPriorReduxPipeline ---")
    repo_redux = "black-forest-labs/FLUX.1-Redux-dev" 
    garment_image_path = "/teamspace/studios/this_studio/.porting/imgs/input_img_1.jpg"
    garment_image_pil = Image.open(garment_image_path).convert("RGB")

    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, torch_dtype=dtype).to(device)
    
    pipe_prior_output = pipe_prior_redux(
        image=garment_image_pil,
        # prompt_embeds=prompt_embeds,
        # pooled_prompt_embeds=pooled_prompt_embeds,
        # generator=torch.Generator(device).manual_seed(123)
    )
    print("pipe_prior_output:\n", pipe_prior_output)
    # # prior_latents = pipe_prior_output.latents
    # print(f"Generated prior latents with shape: {}")

    print("\n--- Step 3: Running the final FluxFillPipeline for inpainting ---")
    image_concat_path = "/teamspace/studios/this_studio/.porting/imgs/img_pixels_concat.jpg"
    mask_path = "/teamspace/studios/this_studio/.porting/imgs/img_mask_concat.jpg"
    image_concat_pil = Image.open(image_concat_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")

    pipe_fill = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", 
        torch_dtype=dtype
    ).to(device)

    output_image = pipe_fill(
        image=image_concat_pil,
        mask_image=mask_pil,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        guidance_scale=7.0,
        num_inference_steps=40,
        strength=0.95,
        generator=torch.Generator(device).manual_seed(456)
    ).images[0]

    output_image.save("final_inpainted_output.png")
    print("\nDone! Final image saved to final_inpainted_output.png")