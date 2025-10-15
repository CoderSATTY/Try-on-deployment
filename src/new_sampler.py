import torch
import joblib
from PIL import Image
import numpy as np
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from torchvision import transforms
from diffusers.pipelines.flux.pipeline_output import FluxPriorReduxPipelineOutput
from utils import zero_out, apply_flux_guidance
from apply_clip import run_clip
from apply_style import CLIPOutputWrapper, load_style_model, apply_stylemodel, STYLE_MODEL_PATH
from tea_cache import teacache_forward

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
    
    prompt_embeds = torch.sum(prompt_embeds, dim=0, keepdim=True)
    pooled_prompt_embeds = torch.sum(pooled_prompt_embeds, dim=0, keepdim=True)
    return prompt_embeds.to(dtype=torch.bfloat16), pooled_prompt_embeds.to(dtype=torch.bfloat16)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    print(f"Using device: {device} with dtype: {dtype}")

    conditioning_cache_path = "/teamspace/studios/this_studio/models/conditioning/text_cache.conditioning"
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
    styled_conds,_,_ = apply_stylemodel(conditioning, style_model, clip_vis_output, 1, "multiply")
    print(f"Shape of styled_conds tensor: {styled_conds[0][0].shape}")

    negative_conds, = zero_out(styled_conds)
    positive_conds, = apply_flux_guidance(styled_conds, 40.3)
    print(f"Shape of negative_conds tensor: {negative_conds[0][0].shape}")
    print(f"Shape of positive_conds tensor: {positive_conds[0][0].shape}")
    
    my_prompt_embeds, my_pooled_prompt_embeds = prepare_embeddings_for_diffusers(
        positive_conds, negative_conds
    )
    print("\nCustom embeddings prepared successfully.")
    print(f"Final prompt_embeds shape for pipeline: {my_prompt_embeds.shape}")
    print(f"Final pooled_prompt_embeds shape for pipeline: {my_pooled_prompt_embeds.shape}")

    print("\nManually creating the prior pipeline output object...")
    prior_redux_output = FluxPriorReduxPipelineOutput(
        prompt_embeds=my_prompt_embeds,
        pooled_prompt_embeds=my_pooled_prompt_embeds
    )
    print("Object created successfully.")

    image_concat_path = "/teamspace/studios/this_studio/Try-on-porting/imgs/img_pixels_concat.jpg"
    mask_path = "/teamspace/studios/this_studio/Try-on-porting/imgs/img_mask_concat.jpg"
    image_concat_pil = Image.open(image_concat_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")
    
    print(f"\nLoaded input image with size: {image_concat_pil.size}")
    print(f"Loaded mask image with size: {mask_pil.size}")


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image_concat_pil).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = transform(mask_pil).unsqueeze(0).to(device, dtype=dtype)

    print("image_tensor data type: ", image_tensor.dtype)
    print("mask_tensor data type: ", mask_tensor.dtype)
    print("prompt_embeds data type: ", my_prompt_embeds.dtype)
    print("pooled_prompt_embeds data type: ", my_pooled_prompt_embeds.dtype)
    print(f"Converted image to tensor with shape: {image_tensor.shape} and dtype: {image_tensor.dtype}")
    print(f"Converted mask to tensor with shape: {mask_tensor.shape} and dtype: {mask_tensor.dtype}")


    pipe_fill = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe_fill.to(device, dtype=dtype)

    # lora_path = "/teamspace/studios/this_studio/models/loras/comfyui_subject_lora16.safetensors"
    # lora_scale = 0.3
    
    # pipe_fill.load_lora_weights(lora_path)
    # print(f"Successfully loaded LoRA from: {lora_path}")

    print("Applying TeaCache optimizations...")
    FluxTransformer2DModel.forward = teacache_forward
    # print("enabling cpu offload")
    # pipe_fill.enable_model_cpu_offload()
    # print("finished cpu offloading")
    num_inference_steps = 50
    print("Setting teacache parameters...")
    pipe_fill.transformer.__class__.enable_teacache = True
    pipe_fill.transformer.__class__.cnt = 0
    pipe_fill.transformer.__class__.num_steps = num_inference_steps
    pipe_fill.transformer.__class__.rel_l1_thresh = 0.4 # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    pipe_fill.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe_fill.transformer.__class__.previous_modulated_input = None
    pipe_fill.transformer.__class__.previous_residual = None
    print("\nRunning FluxFillPipeline...")
    output_image = pipe_fill(
        image=image_tensor,
        mask_image=mask_tensor,
        guidance_scale=50.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device).manual_seed(456),
        # joint_attention_kwargs={"scale": lora_scale},
        **prior_redux_output
    ).images[0]

    output_image.save("final_inpainted_output.png")
    print("\nDone! Final image saved to final_inpainted_output.png")

    
