# import
import time
import sys 
sys.path.append("..") 
from trainers import Pic_HS_trainer
from supar import Parser
import numpy as np
import os, re
import random
from transformers import AutoProcessor, AutoModel
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from pathlib import Path
import math
import heapq
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class Pic_HS_LLM_trainer(Pic_HS_trainer.Pic_HS_trainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type, pic_gen_seed):
        super(Pic_HS_LLM_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type, pic_gen_seed)
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
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(pic_gen_seed)

    def gpt_rephraser(self, sentence):
        api_keys = ['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', 
                    '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',
                    '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', 
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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'] 
        proxy = {
            'http': 'http://localhost:7890',
            'https': 'http://localhost:7890'
        }
        # 设置代理
        openai.proxy = proxy  
        # 定义提示
        prompt = f"Generate a random sentence piece with similar structures as: '{sentence}'"       
        for key in api_keys:
            try:
                # 设置当前API key
                openai.api_key = key              
                # 调用API
                response = openai.Completion.create(
                    engine='gpt-3.5-turbo-instruct',
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.7,
                    n=1,
                    stop=None,
                    timeout=100
                )               
                # 如果成功，从响应中提取改写的句子并返回
                rephrased_sentence = response.choices[0].text.strip()
                return rephrased_sentence
            
            except Exception as e:
                # 如果调用失败，打印错误信息并继续尝试下一个API key
                # print(f"Error with API key {key}: {e}")
                continue
        
        # 如果所有API keys都尝试失败，返回错误信息
        return "Failed to rephrase the sentence using all available API keys."


    def gpt_adjuster(self, sentence):
        api_keys = ['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', 
                    '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',
                    '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', 
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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',
                    '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'] 
        proxy = {
            'http': 'http://localhost:7890',
            'https': 'http://localhost:7890'
        }
        # 设置代理
        openai.proxy = proxy  
        # Define the prompt
        prompt = f"Rephrase and refine the following piece of sentence in more detail: '{sentence}'"
        # Generate text using the completions API
        for key in api_keys:
            try:
                # 设置当前API key
                openai.api_key = key              
                # 调用API
                response = openai.Completion.create(
                    engine='gpt-3.5-turbo-instruct',
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.7,
                    n=1,
                    stop=None,
                    timeout=100
                )               
                # 如果成功，从响应中提取改写的句子并返回
                adjusted_sentence = response.choices[0].text.strip()
                return adjusted_sentence
            
            except Exception as e:
                # 如果调用失败，打印错误信息并继续尝试下一个API key
                # print(f"Error with API key {key}: {e}")
                continue
        
        # 如果所有API keys都尝试失败，返回错误信息
        return "Failed to rephrase the sentence using all available API keys."


    # def mutated(self, candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args):
    #         # print("Running LLM mutation")
    #         deleted = {}
    #         added = {}
    #         prompt_count = args.num_candidates
    #         candidates = []
    #         for i in range (0, prompt_count):
    #             # rephrase the sentence
    #             candidate = self.gpt_rephraser(candidate)
    #             candidates.append(candidate)
    #             # time.sleep(10)
    #             i = i + 1
    #         return candidates, deleted, added

    def mutated(self, candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args):
        # print("Running LLM mutation")
        deleted = {}
        added = {}
        candidate = self.gpt_rephraser(candidate)
        return candidate, deleted, added



            
