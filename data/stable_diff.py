from diffusers import StableDiffusionPipeline
import torch

def prompt_download(prompts,location_prompt):
   try:
        # print("reading propmts" + prompts)
        prompt_list = []
        for prompt in prompts:
            print("reading prompts at " + str(location_prompt) + str(prompt))
            f = open(location_prompt + prompt, 'r')
            text = f.read().replace('\n', ' ').replace('\r', ' ')
            prompt_list.append(text)


        if  len(prompt_list) > 0:
            return prompt_list
   except:
       print("failed to read prompts")

def stable_default():
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    images = []
    location = 'prompts/'
    prompt_text = ['prompt_0.txt', 'prompt_1.txt', 'prompt_2.txt', 'prompt_3.txt', 'prompt_4.txt']
    prompts = prompt_download(prompt_text, location)

    for i, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        image.save(f'output/result_{i}.jpg')
        images.append(image)


if __name__ == "__main__":
    stable_default()


