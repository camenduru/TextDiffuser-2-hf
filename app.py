import os
import re
import zipfile
import torch
import gradio as gr
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel 
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

import string
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
'''alphabet
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
'''

# if not os.path.exists('Arial.ttf'):
#     os.system('wget https://huggingface.co/datasets/JingyeChen22/TextDiffuser/resolve/main/Arial.ttf')


if not os.path.exists('images2'):
    os.system('wget https://huggingface.co/datasets/JingyeChen22/TextDiffuser/resolve/main/images2.zip')
    with zipfile.ZipFile('images2.zip', 'r') as zip_ref:
        zip_ref.extractall('.')


# if not os.path.exists('architecture.jpg'):
# os.system('wget https://huggingface.co/JingyeChen22/textdiffuser2-full-ft/tree/main/layout_planner_m1')

# if not os.path.exists('gray256.jpg'):
#     os.system('wget https://huggingface.co/JingyeChen22/textdiffuser2-full-ft/blob/main/gray256.jpg')

# print(os.system('apt install mlocate'))
os.system('ls')
# print(os.system('pwd'))
# print(os.system('locate gray256.jpg'))
# # img = Image.open('locate gray256.jpg')
# # print(img.size)
# exit(0)

#### import m1
from fastchat.model import load_model, get_conversation_template
m1_model_path = 'JingyeChen22/textdiffuser2_layout_planner'
m1_model, m1_tokenizer = load_model(
    m1_model_path,
    'cuda',
    1,
    None,
    False,
    False,
    revision="main",
    debug=False,
)

#### import diffusion models
text_encoder = CLIPTextModel.from_pretrained(
    'JingyeChen22/textdiffuser2-full-ft', subfolder="text_encoder", ignore_mismatched_sizes=True
).cuda()
tokenizer = CLIPTokenizer.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder="tokenizer"
)

#### additional tokens are introduced, including coordinate tokens and character tokens
print('***************')
print(len(tokenizer))
for i in range(520):
    tokenizer.add_tokens(['l' + str(i) ]) # left
    tokenizer.add_tokens(['t' + str(i) ]) # top
    tokenizer.add_tokens(['r' + str(i) ]) # width
    tokenizer.add_tokens(['b' + str(i) ]) # height    
for c in alphabet:
    tokenizer.add_tokens([f'[{c}]']) 
print(len(tokenizer))
print('***************')

vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae").cuda()
unet = UNet2DConditionModel.from_pretrained(
    'JingyeChen22/textdiffuser2-full-ft', subfolder="unet"
).cuda()
text_encoder.resize_token_embeddings(len(tokenizer))


#### for interactive
stack = []
state = 0   
font = ImageFont.truetype("./Arial.ttf", 32)

def skip_fun(i, t):
    global state
    state = 0


def exe_undo(i, t):
    global stack
    global state
    state = 0
    stack = []
    image = Image.open(f'./gray256.jpg')
    print('stack', stack)
    return image


def exe_redo(i, t):
    global state 
    state = 0

    if len(stack) > 0:
        stack.pop()
    image = Image.open(f'./gray256.jpg')
    draw = ImageDraw.Draw(image)

    for items in stack:
        # print('now', items)
        text_position, t = items
        if len(text_position) == 2:
            x, y = text_position
            text_color = (255, 0, 0)  
            draw.text((x+2, y), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
        elif len(text_position) == 4:
            x0, y0, x1, y1 = text_position
            text_color = (255, 0, 0)  
            draw.text((x0+2, y0), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x0-r, y0-r)
            rightDownPoint = (x0+r, y0+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
            draw.rectangle((x0,y0,x1,y1), outline=(255, 0, 0) )

    print('stack', stack)
    return image

def get_pixels(i, t, evt: gr.SelectData):
    global state

    text_position = evt.index

    if state == 0:
        stack.append(
            (text_position, t)
        )
        print(text_position, stack)
        state = 1
    else:
        
        (_, t) = stack.pop()
        x, y = _
        stack.append(
            ((x,y,text_position[0],text_position[1]), t)
        )
        state = 0


    image = Image.open(f'./gray256.jpg')
    draw = ImageDraw.Draw(image)

    for items in stack:
        # print('now', items)
        text_position, t = items
        if len(text_position) == 2:
            x, y = text_position
            text_color = (255, 0, 0)  
            draw.text((x+2, y), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
        elif len(text_position) == 4:
            x0, y0, x1, y1 = text_position
            text_color = (255, 0, 0)  
            draw.text((x0+2, y0), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x0-r, y0-r)
            rightDownPoint = (x0+r, y0+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
            draw.rectangle((x0,y0,x1,y1), outline=(255, 0, 0) )

    print('stack', stack)

    return image




def text_to_image(prompt,keywords,slider_step,slider_guidance,slider_batch,slider_temperature,slider_natural):

    global stack
    global state

    with torch.no_grad():
        time1 = time.time()
        user_prompt = prompt

        if slider_natural:
            user_prompt += ' <|endoftext|>'
            composed_prompt = tokenizer.decode(prompt)
        else:
            if len(stack) == 0:

                if len(keywords.strip()) == 0:
                    template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. All keywords are included in the caption. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {user_prompt}'
                else:
                    keywords = keywords.split('/')
                    keywords = [i.strip() for i in keywords]
                    template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. In addition, we also provide all keywords at random order for reference. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {prompt}. Keywords: {str(keywords)}'

                msg = template
                conv = get_conversation_template(m1_model_path)
                conv.append_message(conv.roles[0], msg)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                inputs = m1_tokenizer([prompt], return_token_type_ids=False)
                inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
                output_ids = m1_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=slider_temperature,
                    repetition_penalty=1.0,
                    max_new_tokens=512,
                )

                if m1_model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
                outputs = m1_tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )
                print(f"[{conv.roles[0]}]\n{msg}")
                print(f"[{conv.roles[1]}]\n{outputs}")
                ocrs = outputs.split('\n')
                time2 = time.time()
                print(time2-time1)
                
                # user_prompt = prompt
                current_ocr = ocrs

                ocr_ids = [] 
                print('user_prompt', user_prompt)
                print('current_ocr', current_ocr)
                

                for ocr in current_ocr:
                    ocr = ocr.strip()

                    if len(ocr) == 0 or '###' in ocr or '.com' in ocr:
                        continue

                    items = ocr.split()
                    pred = ' '.join(items[:-1])
                    box = items[-1]
                
                    l,t,r,b = box.split(',')
                    l,t,r,b = int(l), int(t), int(r), int(b)
                    ocr_ids.extend(['l'+str(l), 't'+str(t), 'r'+str(r), 'b'+str(b)])

                    char_list = list(pred)
                    char_list = [f'[{i}]' for i in char_list]
                    ocr_ids.extend(char_list)
                    ocr_ids.append(tokenizer.eos_token_id)     

                caption_ids = tokenizer(
                    user_prompt, truncation=True, return_tensors="pt"
                ).input_ids[0].tolist() 

                try:
                    ocr_ids = tokenizer.encode(ocr_ids)
                    prompt = caption_ids + ocr_ids
                except:
                    prompt = caption_ids

                composed_prompt = tokenizer.decode(prompt)
            
            else:
                user_prompt += ' <|endoftext|>'

                
                for items in stack:
                    position, text = items

                    
                    if len(position) == 2:
                        x, y = position
                        x = x // 4
                        y = y // 4
                        text_str = ' '.join([f'[{c}]' for c in list(text)])
                        user_prompt += f'<|startoftext|> l{x} t{y} {text_str} <|endoftext|>'
                    elif len(position) == 4:
                        x0, y0, x1, y1 = position
                        x0 = x0 // 4
                        y0 = y0 // 4
                        x1 = x1 // 4
                        y1 = y1 // 4
                        text_str = ' '.join([f'[{c}]' for c in list(text)])
                        user_prompt += f'<|startoftext|> l{x0} t{y0} r{x1} b{y1} {text_str} <|endoftext|>'

                    composed_prompt = user_prompt
                    prompt = tokenizer.encode(user_prompt)

        prompt = prompt[:77]
        while len(prompt) < 77: 
            prompt.append(tokenizer.pad_token_id) 
        prompts_cond = prompt
        prompts_nocond = [tokenizer.pad_token_id]*77

        prompts_cond = [prompts_cond] * slider_batch
        prompts_nocond = [prompts_nocond] * slider_batch

        prompts_cond = torch.Tensor(prompts_cond).long().cuda()
        prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()

        scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler") 
        scheduler.set_timesteps(slider_step) 
        noise = torch.randn((slider_batch, 4, 64, 64)).to("cuda") 
        input = noise

        encoder_hidden_states_cond = text_encoder(prompts_cond)[0]
        encoder_hidden_states_nocond = text_encoder(prompts_nocond)[0] 


        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():  # classifier free guidance
                noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:slider_batch]).sample # b, 4, 64, 64
                noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:slider_batch]).sample # b, 4, 64, 64
                noisy_residual = noise_pred_uncond + slider_guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                input = prev_noisy_sample

        # decode
        input = 1 / vae.config.scaling_factor * input 
        images = vae.decode(input, return_dict=False)[0] 
        width, height = 512, 512
        results = []
        new_image = Image.new('RGB', (2*width, 2*height))
        for index, image in enumerate(images.float()):
            image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
            results.append(image)
            row = index // 2
            col = index % 2
            new_image.paste(image, (col*width, row*height))
        # new_image.save(f'{args.output_dir}/pred_img_{sample_index}_{args.local_rank}.jpg')
        # results.insert(0, new_image)
        # return new_image
        os.system('nvidia-smi')
        return tuple(results),  composed_prompt
    
with gr.Blocks() as demo:

    gr.HTML(
        """
        <div style="text-align: center; max-width: 1600px; margin: 20px auto;">
        <h2 style="font-weight: 900; font-size: 2.5rem; margin: 0rem">
            TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
        </h2>
        <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem"> 
        [<a href="https://arxiv.org/abs/2311.16465" style="color:blue;">arXiv</a>] 
        [<a href="https://github.com/microsoft/unilm/tree/master/textdiffuser-2" style="color:blue;">Code</a>]
        </h3> 
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        We propose <b>TextDiffuser-2</b>, aiming at unleashing the power of language models for text rendering. Specifically, we <b>tame a language model into a layout planner</b> to transform user prompt into a layout using the caption-OCR pairs. The language model demonstrates flexibility and automation by inferring keywords from user prompts or incorporating user-specified keywords to determine their positions. Secondly, we <b>leverage the language model in the diffusion model as the layout encoder</b> to represent the position and content of text at the line level. This approach enables diffusion models to generate text images with broader diversity.
        </h2>
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        👀 <b>Tips for using this demo</b>: <b>(1)</b> Please carefully read the disclaimer in the below. <b>(2)</b> The specification of keywords is optional. If provided, the language model will do its best to plan layouts using the given keywords. <b>(3)</b> If a template is given, the layout planner (M1) is not used. <b>(4)</b> Three operations, including redo, undo, and skip are provided. When using skip, only the left-top point of a keyword will be recorded, resulting in more diversity but sometimes decreasing the accuracy. <b>(5)</b> The layout planner can produce different layouts. You can increase the temperature to enhance the diversity.
        </h2>

        <style>
            .scaled-image {
                transform: scale(1);
            }
        </style>
        
        <img src="https://i.ibb.co/vmrXRb5/architecture.jpg" alt="textdiffuser-2" class="scaled-image">
        </div>
        """)

    with gr.Tab("Text-to-Image"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Input your prompt here.", placeholder="A beautiful city skyline stamp of Shanghai")
                keywords = gr.Textbox(label="(Optional) Input your keywords here. Keywords should bu seperate by / (e.g., keyword1/keyword2/...)", placeholder="keyword1/keyword2")

                # 这里加一个会话框
                with gr.Row():
                    with gr.Column(scale=1):
                        i = gr.Image(label="Template (Click to paint)", type='filepath', value=f'./gray256.jpg', height=256, width=256)
                    with gr.Column(scale=1):
                        t = gr.Textbox(label="Keyword", value='input_keyword')
                        redo = gr.Button(value='Redo - Cancel the last keyword') # 如何给b绑定事件
                        undo = gr.Button(value='Undo - Clear the canvas') # 如何给b绑定事件
                        skip_button = gr.Button(value='Skip - Operate next keyword') # 如何给b绑定事件

                i.select(get_pixels,[i,t],[i])
                redo.click(exe_redo, [i,t],[i])
                undo.click(exe_undo, [i,t],[i])
                skip_button.click(skip_fun, [i,t])

                # radio = gr.Radio(["Stable Diffusion v2.1", "Stable Diffusion v1.5"], label="Pre-trained Model", value="Stable Diffusion v1.5")
                slider_step = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Sampling step", info="The sampling step for TextDiffuser.")
                slider_guidance = gr.Slider(minimum=1, maximum=9, value=7.5, step=0.5, label="Scale of classifier-free guidance", info="The scale of classifier-free guidance and is set to 7.5 in default.")
                slider_batch = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Batch size", info="The number of images to be sampled.")
                slider_temperature = gr.Slider(minimum=0.1, maximum=2, value=0.7, step=0.1, label="Temperature", info="Control the diversity of layout planner. Higher value indicates more diversity.")
                slider_natural = gr.Checkbox(label="Natural image generation", bool=False, info="The text position and content info will not be incorporated.")
                slider_seed = gr.Slider(minimum=1, maximum=10000, label="Seed", randomize=True)
                button = gr.Button("Generate")
                            
            with gr.Column(scale=1):
                output = gr.Gallery(label='Generated image')

                with gr.Accordion("Intermediate results", open=False):
                    gr.Markdown("Composed prompt")
                    composed_prompt = gr.Textbox(label='')
                
                # with gr.Accordion("Intermediate results", open=False):
                #     gr.Markdown("Layout, segmentation mask, and details of segmentation mask from left to right.")
                #     intermediate_results = gr.Image(label='')
        
        # gr.Markdown("## Prompt Examples")

        button.click(text_to_image, inputs=[prompt,keywords,slider_step,slider_guidance,slider_batch,slider_temperature,slider_natural], outputs=[output, composed_prompt])

        gr.Markdown("## Prompt Examples")
        gr.Examples(
            [
                ["A beautiful city skyline stamp of Shanghai", ""],
                ["A book cover named summer vibe", ""],
            ],
            prompt,
            keywords,
            examples_per_page=10
        )



    gr.HTML(
        """
        <div style="text-align: justify; max-width: 1100px; margin: 20px auto;">
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Version</b>: 1.0
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Contact</b>: 
        For help or issues using TextDiffuser-2, please email Jingye Chen <a href="mailto:qwerty.chen@connect.ust.hk">(qwerty.chen@connect.ust.hk)</a>, Yupan Huang <a href="mailto:huangyp28@mail2.sysu.edu.cn">(huangyp28@mail2.sysu.edu.cn)</a> or submit a GitHub issue. For other communications related to TextDiffuser-2, please contact Lei Cui <a href="mailto:lecu@microsoft.com">(lecu@microsoft.com)</a> or Furu Wei <a href="mailto:fuwei@microsoft.com">(fuwei@microsoft.com)</a>.
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Disclaimer</b>: 
        Please note that the demo is intended for academic and research purposes <b>ONLY</b>. Any use of the demo for generating inappropriate content is strictly prohibited. The responsibility for any misuse or inappropriate use of the demo lies solely with the users who generated such content, and this demo shall not be held liable for any such use.
        </h3>
        </div>
        """
    )


demo.launch()