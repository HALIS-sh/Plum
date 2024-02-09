from transformers import AutoProcessor, AutoModel
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

original_candidate = "a photo of an astronaut riding a horse on mars"
original_score = 0.0
# load model
device = "cuda"
# self.processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# self.model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
# self.sd_model_id = "stabilityai/stable-diffusion-2-1"
processor = AutoProcessor.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K")
model = AutoModel.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--yuvalkirstain--PickScore_v1").eval().to(device)


# prepare prompt_0 and base_prompt_pics
def initialize_prompt_0():
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    sd_pipe = StableDiffusionPipeline.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6", torch_dtype=torch.float16)
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to("cuda")
    pic_count = 1
    k = 2
    for pic_count in range(1,k+1):
        image = sd_pipe(original_candidate).images[0]
        file_name = "prompt_{}_images_{}.png".format(0, pic_count)
        image.save(file_name)
        pic_count = pic_count + 1

def calc_probs(prompt, images):
    
    # preprocess
    # max_length = 77 check if this is the right max_length ###############
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=12,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

def pics_score(candidate, prompt_count):

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    # sd_pipe = StableDiffusionPipeline.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6", torch_dtype=torch.float16)
    # sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    # sd_pipe = sd_pipe.to("cuda")
    # prompt_count = 1
    pic_count = 1
    prompt = candidate
    k = 2
    # generate the pics of prompt_n
    # for pic_count in range(1, k + 1):
    #     image = sd_pipe(prompt).images[0]
    #     file_name = "prompt_{}_images_{}.png".format(prompt_count, pic_count)
    #     image.save(file_name)
    #     pic_count = pic_count + 1

    # initialize the pic_count
    pic_count = 1
    score = 0.0
    # calculate the average score of prompt_n
    for pic_count in range(1, k + 1):
        
        file_name_n = "prompt_{}_images_{}.png".format(0, pic_count)
        file_name_0 = "prompt_{}_images_{}.png".format(prompt_count, pic_count)
        pil_images = [Image.open(file_name_0), Image.open(file_name_n)]
        score = score + (calc_probs(original_candidate, pil_images)[1]*100)
        print("score:", score)
    ave_score = score/float(k)
    print("ave_score:", ave_score)
    return ave_score

def main():
    # initialize_prompt_0()
    improved_prompt = "a photo of a strong American astronaut on mars, he is riding a horse."
    pics_score(improved_prompt, 1)

if __name__ == "__main__":
    main()