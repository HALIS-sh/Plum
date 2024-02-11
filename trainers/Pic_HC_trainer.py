# import
from trainers import HC_trainer
from supar import Parser
import numpy as np
import os, re
import random
from transformers import AutoProcessor, AutoModel
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import wandb
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class Pic_HC_trainer(HC_trainer.HC_trainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type):
        super(Pic_HC_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type)
        self.patience_counter = 1
        self.W_candidates = []
        self.W_scores = []
        self.original_candidate = None
        self.original_score = None
        self.result_candidate = None
        self.result_score = None
        self.parser = Parser.load('crf-con-en')
        self.para_tokenizer = None
        self.para_model = None
        self.state = {}

        # load model
        self.device = "cuda"
        # self.processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        # self.model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        # self.sd_model_id = "stabilityai/stable-diffusion-2-1"
        self.processor = AutoProcessor.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--yuvalkirstain--PickScore_v1").eval().to(self.device)
        
    # prepare prompt_0 and base_prompt_pics
    def initialize_prompt_0(self, args):
        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        sd_pipe = StableDiffusionPipeline.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6", torch_dtype=torch.float16)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to("cuda")
        # self.original_candidate = "Show the boundary between night and day."
        self.original_candidate = args.original_candidate
        self.original_score = 50.0
        self.result_candidate = self.original_candidate
        self.result_score = self.original_score
        pic_count = 1
        folder_name = args.meta_pic_dir
        k = args.pics_number
        for pic_count in range(1,k+1):
            image = sd_pipe(self.original_candidate).images[0]
            file_name = "{}/prompt_{}_images_{}.png".format(folder_name, 0 , pic_count)
            image.save(file_name)
            pic_count = pic_count + 1

    def calc_probs(self, prompt, images):
        
        # preprocess
        # max_length = 77 check if this is the right max_length ###############
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=0.5,
            return_tensors="pt",
        ).to(self.device)
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=12,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
        
        return probs.cpu().tolist()


    def score(self, candidate, prompt_count, split='train', write=False, args=None):

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        sd_pipe = StableDiffusionPipeline.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6", torch_dtype=torch.float16)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to("cuda")
        # prompt_count = 1
        pic_count = 1
        prompt = candidate
        folder_name = args.meta_pic_dir
        k = args.pics_number
        # # generate the pics of prompt_n
        for pic_count in range(1, k + 1):
            image = sd_pipe(prompt).images[0]
            file_name = "{}/prompt_{}_images_{}.png".format(folder_name,prompt_count, pic_count)
            image.save(file_name)
            pic_count = pic_count + 1

        # initialize the pic_count and score
        pic_count = 1
        score = 0.0

        # calculate the average score of prompt_n
        for pic_count in range(1, k + 1):
            file_name_0 = "{}/prompt_{}_images_{}.png".format(folder_name, 0, pic_count)
            file_name_n = "{}/prompt_{}_images_{}.png".format(folder_name, prompt_count, pic_count)
            pil_images = [Image.open(file_name_0), Image.open(file_name_n)]
            score = score + (self.calc_probs(self.original_candidate, pil_images)[1]*100)
        ave_score = score/float(k)
        # print("ave_score:", ave_score)
        return ave_score
