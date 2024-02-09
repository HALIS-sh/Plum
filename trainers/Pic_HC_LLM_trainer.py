# import
import time
from trainers import Pic_HC_trainer
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


class Pic_HC_LLM_trainer(Pic_HC_trainer.Pic_HC_trainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type):
        super(Pic_HC_LLM_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type)
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
        
    

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def gpt_rephraser(self, sentence, current_iteration):
        proxy = {
        'http': 'http://localhost:7890',
        'https': 'http://localhost:7890'
        }
        api_keys = ['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', 
                    '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',
                    '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', 
                    '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']
        openai.proxy = proxy
        # Set up your OpenAI API credentials
        openai.api_key = api_keys[current_iteration%len(api_keys)]
        # Define the prompt
        prompt = f"Rephrase and refine the following sentence in more detail : '{sentence}'"
        # Generate text using the completions API
        response = openai.Completion.create(
            engine='gpt-3.5-turbo-instruct',  # You can also use 'gpt-3.5-turbo' for faster responses
            prompt=prompt,
            max_tokens=50,  # You can adjust this based on the desired length of the response
            temperature=0.7,  # Controls the randomness of the output, lower values make it more focused
            n=1,  # Generate only one response
            stop=None,  # Let the model generate a full response without any specific stop sequence
            timeout = 100
        )

        # Extract the rephrased sentence from the API response
        rephrased_sentence = response.choices[0].text.strip()

        return rephrased_sentence


    def mutated(self, candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args, current_iteration):
        deleted = {}
        added = {}
        prompt_count = args.num_candidates
        candidates = []
        scores = []
        for i in range (0, prompt_count):
            # rephrase the sentence
            candidate = self.gpt_rephraser(candidate, current_iteration)
            candidates.append(candidate)
            time.sleep(10)
            i = i + 1
        for c, candidate in enumerate(candidates):
            scores.append(self.score(candidate, c+1, args=args))
            print("Candidate: ", candidate)
            print("Ave_Score: ", scores[-1])
        return candidates, scores, deleted, added
