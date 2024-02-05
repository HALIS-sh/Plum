# import
import sys 
sys.path.append("..") 
from trainers.base_trainer import SimpleTrainer
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


class Pic_HS_trainer(SimpleTrainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates):
        super(Pic_HS_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates)
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
    def initialize_prompt_0(self):
        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        sd_pipe = StableDiffusionPipeline.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6", torch_dtype=torch.float16)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to("cuda")
        pic_count = 1
        self.original_candidate = "a photo of an astronaut riding a horse on mars"
        self.original_score = 0.0
        self.result_candidate = self.original_candidate
        self.result_score = self.original_score
        for pic_count in range(1,3):
            image = sd_pipe(self.original_candidate).images[0]
            file_name = "prompt_{}_images_{}.png".format(0, pic_count)
            image.save(file_name)
            pic_count = pic_count + 1


    def get_state(self, current_iteration, delete_tracker):
        self.state = {'np_random_state' : np.random.get_state(), 'random_state' : random.getstate(), 'current_iteration' : current_iteration, 'W_candidates' : self.W_candidates, 'W_scores' : self.W_scores, 'result_candidate' : self.result_candidate, 'result_score' : self.result_score, 'patience_counter': self.patience_counter, 'delete_tracker' : delete_tracker}
        
    def set_state(self):
        current_iteration = self.state['current_iteration'] 
        delete_tracker = self.state['delete_tracker']
        self.W_candidates = self.state['W_candidates'] 
        self.W_scores = self.state['W_scores'] 
        self.result_candidate = self.state['result_candidate']
        self.result_score =  self.state['result_score']
        self.patience_counter = self.state['patience_counter']
        np.random.set_state(self.state['np_random_state'])
        random.setstate(self.state['random_state'])
        return current_iteration, delete_tracker

    def calc_probs(self, prompt, images):
        
        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
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

    def containenglish(self, str0):
        return bool(re.search('[a-z A-Z]', str0))

    # mutate the prompt
    def mutated(self, base_candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args):

        deleted = {}
        added = {}
        
        if base_candidate == self.original_candidate:
            for p in phrase_lookup.values(): print(p)
        if use_add: 
            if len(delete_tracker): 
                if 'add' not in edit_operations: edit_operations.append('add')
            else: 
                if 'add' in edit_operations: edit_operations.remove('add')

        empty = True
        while empty:
            if self.num_compose == 1:
                edits = np.random.choice(edit_operations, self.num_candidates)
            else: 
                edits = []
                for n in range(self.num_candidates):
                    edits.append(np.random.choice(edit_operations, self.num_compose))
            print(edits)

        # generate candidates
            candidates = []
            for edit in edits:
                if isinstance(edit, str): 
                    print("base_candidate:", base_candidate)
                    candidate, indices = self.perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                    empty = not self.containenglish(candidate)
                    if not empty:
                        print(candidate)
                        candidates.append(candidate)
                        if edit  == 'del': deleted[candidate] = [phrase_lookup[indices[0]]]
                        if edit == 'add': 
                            if len(indices): added[candidate] = indices
                    else:
                        print('''Note: The mutated candidate is an empty string, and it is deleted.''')
                else:
                    old_candidate = base_candidate
                    composed_deletes = []
                    composed_adds = []
                    for op in edit:
                        phrase_lookup = self.get_phrase_lookup(old_candidate, args)
                        new_candidate, indices = self.perform_edit(op, old_candidate, phrase_lookup, delete_tracker)
                        empty = not self.containenglish(new_candidate)
                        if not empty:
                            print(new_candidate)
                            if op  == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                            if op == 'add': 
                                if len(indices): composed_adds.append(indices[0])
                            old_candidate = new_candidate
                        else:
                            break

                    if not empty:
                        candidates.append(new_candidate)
                        if 'del' in edit: deleted[new_candidate] = composed_deletes
                        if 'add' in edit and len(composed_adds) > 0: added[new_candidate] = composed_adds
        # scores = []
        # for c, candidate in enumerate(candidates):
        #     scores.append(self.pics_score(candidate, c, args=args))
        #     print(scores[-1])

        return candidates, deleted, added 


    def pics_score(self, candidate, prompt_count, args):
        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        sd_pipe = StableDiffusionPipeline.from_pretrained("/home/wenhesun/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6", torch_dtype=torch.float16)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to("cuda")
        # prompt_count = 1
        pic_count = 1
        prompt = candidate
        k = 2

        # generate the pics of prompt_n
        for pic_count in range(1, k+1):
            image = sd_pipe(prompt).images[0]
            file_name = "prompt_{}_images_{}.png".format(prompt_count, pic_count)
            image.save(file_name)
            pic_count = pic_count + 1

        # initialize the pic_count
        pic_count = 1

        # calculate the average score of prompt_n
        for pic_count in range(1, k+1):
            score = 0.0
            file_name_0 = "prompt_{}_images_{}.png".format(0, pic_count)
            file_name_n = "prompt_{}_images_{}.png".format(prompt_count, pic_count)
            pil_images = [Image.open(file_name_0), Image.open(file_name_n)]
            score = score + (self.calc_probs(self.original_candidate, pil_images)[1]*100)
        ave_score = score/float(k)
        
        print("ave_score:", ave_score)
        return ave_score

    def generate_candidate(self, ks, HMCR, PAR, edit_opertions_small, use_add, delete_tracker, edit_operations, args):
    
        w_m=[]
        w_m_words = []

        for j in range(ks):
            print("length of W_candidates: ", len(self.W_candidates))
            idx = np.random.randint(0,len(self.W_candidates))
            w = self.W_candidates[idx]
            phrases_pun = self.get_phrase_lookup_pun(w, args)
            w_phrases = list(phrases_pun.values())
            L = len(w_phrases)
            print(w_phrases)
            print("L: ", L)
            start = math.ceil(j/ks*L)
            print("start: ", start)
            end = math.ceil((j+1)/ks*L) - 1
            print("end: ", end)
            if(start!=end):
                w_segement = w_phrases[start:end]
            else:
                w_segement = w_phrases[start]
            w_segement_words = []

            for phrase in w_segement:
                w_segement_words = w_segement_words + [phrase]
            w_segement = w_segement_words
            w_segement = str(w_segement)
            if HMCR >= np.random.random():
                if PAR >= np.random.random():
                    try:
                        print("w_segement:", w_segement)
                        phrase_lookup = self.get_phrase_lookup(w_segement, args)
                        candidate, _ = self.perform_edit(edit_opertions_small, w_segement, phrase_lookup, delete_tracker)
                        w_segement = candidate
                        print(w_segement)
                    except:
                        print('Error occurs (parser) and skip this mutation')
                        continue
                deleted = {}
                added = {}
                   
            else:
                try:
                    print("w_segement:", w_segement)
                    phrase_lookup = self.get_phrase_lookup(w_segement, args)
                    candidates, deleted, added = self.mutated(w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                    w_segement = candidates[0] # multipule edit operations can be implemented if necessary
                    print(w_segement)
                except:
                    print('Error occurs (parser) and skip this mutation')
                    continue
                
            w_m.append(w_segement)
        for segement in w_m:
            w_m_words.extend(self.word_tokenize(segement))
        w_m = self.detokenize(w_m_words)
        return w_m, deleted, added

    def train(self, args):

        ks = 1
        HMCR = 0.4
        PAR = 0.5
        edit_opertions_small = 'sub'
        N_H = 10

        current_iteration = 0
        delete_tracker = []
        edit_operations = args.edits
        use_add = 'add' in edit_operations

        if 'sub' in edit_operations:
            self.if_sub(edit_operations)

        self.W_candidates.append(self.original_candidate)

        while current_iteration < self.maxiter:
            current_iteration = current_iteration + 1
            print("Current Iteration: ", current_iteration)
            #Base_candidate after battled in the tournament
            base_candidate = self.result_candidate
            base_score = self.result_score

            # Generate a candidate
            deleted_candidate = {}
            added_candidate = {}
            self.W_candidates_m = []
            self.W_scores_m = []
            
            for c in range(args.num_candidates):
                w_m, deleted, added = self.generate_candidate(ks, HMCR, PAR, edit_opertions_small, use_add, delete_tracker, edit_operations, args)
                self.W_candidates_m.append(w_m)
                self.W_scores_m.append(self.pics_score(w_m, c, args))
                
                deleted_list = []
                added_list = []
                for item in list(deleted.values()):
                    deleted_list.extend(item)
                for item in list(added.values()):
                    added_list.extend(item)
                if not deleted_list == []:
                    deleted_candidate[w_m] = deleted_list
                if not added_list == []:
                    added_candidate[w_m] = added_list

            self.update_result(self.W_candidates_m, self.W_scores_m)

            if self.patience_counter > args.patience:
                print('Ran out of patience')
                break
            self.W_candidates = self.W_candidates + self.W_candidates_m
            self.W_scores = self.W_scores + self.W_scores_m
            
            top_N_H_idx_list = heapq.nlargest(N_H, range(len(self.W_scores)), self.W_scores.__getitem__)
            W_candidates_top_N_H = [self.W_candidates[i] for i in top_N_H_idx_list]
            W_scores_top_N_H = [self.W_scores[i] for i in top_N_H_idx_list]
            self.W_candidates = W_candidates_top_N_H
            self.W_scores = W_scores_top_N_H
            
            
            for candidate in self.W_candidates:

                if candidate in added_candidate.keys():
                    print('Notice! Prev tracker: ', delete_tracker)
                    for chunk in added_candidate[candidate]: 
                        try: 
                            delete_tracker.remove(chunk)
                        except: 
                            pass
                    print('Notice! New tracker: ', delete_tracker)

                if candidate in deleted_candidate.keys():
                    delete_tracker.extend(deleted_candidate[candidate])

                print("result_candidate: ", self.result_candidate)

            
