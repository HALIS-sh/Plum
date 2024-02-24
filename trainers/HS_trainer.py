from trainers.base_trainer import SimpleTrainer
import numpy as np
import os, re
import json
import wandb
from supar import Parser
import sys 
sys.path.append("..") 
import utils.nat_inst_gpt3 as gpt3
import utils.nat_inst_gpt2 as gpt2
import random
from pathlib import Path
import math
import heapq
from transformers import AutoProcessor, AutoModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import openai

class HS_trainer(SimpleTrainer):

    def __init__(self, maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type):
        super(HS_trainer, self).__init__(maxiter, patience, train_seed, seed, num_compose, num_candidates, backbone, task_type)
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
        self.W_candidates_m = []
        self.W_scores_m = []
        
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

    def tournament_selection(self):
        S_candidates = []
        S_scroes = []
        for k in range(self.num_tournaments):  
            parent = np.random.randint(0,len(self.W_candidates))  # parent, score_parent <-- Random(W)
            S_candidates.append(self.W_candidates[parent])  # S_candidate = S_candidate + parent 
            S_scroes.append(self.W_scores[parent])  # S_score = S_score + score_parent
        base_idx = np.argmax(S_scroes)   # base_idx = \arg max_{idx \in S} S_score
        base_candidate = S_candidates[base_idx] # base <-- S_candidates(base_idx)
        base_score = S_scroes[base_idx] # base_score <-- S_candidates(base_idx)
        
        return base_candidate, base_score

    def containenglish(self, str0):
        return bool(re.search('[a-z A-Z]', str0))

    def mutated(self, base_candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args):

        deleted = {}
        added = {}
        
        # if base_candidate == self.original_candidate:
        #     for p in phrase_lookup.values(): print(p)
        if use_add: 
            if len(delete_tracker): 
                if 'add' not in edit_operations: edit_operations.append('add')
            else: 
                if 'add' in edit_operations: edit_operations.remove('add')

        empty = True
        while empty:
            if self.num_compose == 1:
                edits = np.random.choice(edit_operations, 1)
            else: 
                edits = []
                for n in range(1):
                    edits.append(np.random.choice(edit_operations, self.num_compose))
            print(edits)

        # generate candidates
            candidates = []
            for edit in edits:
                if isinstance(edit, str): 
                    candidate, indices = self.perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
                    empty = not self.containenglish(candidate)
                    if not empty:
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

        return candidates, deleted, added 
    
    # def gpt_adjuster(self, sentence, current_iteration):
    #     raise NotImplementedError("Subclasses must implement call_child_method")
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
        prompt = f"Slightly adjust the following sentence: '{sentence}'"
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


    def generate_candidate(self, ks, HMCR, PAR, edit_opertions_small, use_add, delete_tracker, edit_operations, args, current_iteration):
    
        w_m=[]
        w_m_words = []
        if args.use_commas_split:
            print("use_commas_split")
            for j in range (ks):
                idx = np.random.randint(0,len(self.W_candidates))
                w = self.W_candidates[idx]
                parts = w.split(',')
                L = len(parts)
                if L < 5:
                    if L == 1:
                        parts[0] = parts[0] + ','
                    parts = parts * 3
                start = math.ceil(j/ks*L)
                end = math.ceil((j+1)/ks*L) - 1
                if(start!=end):
                    w_segement = parts[start:end]
                else:
                    w_segement = [parts[start]]
                w_segement_words = []
                for segement in w_segement:
                    w_segement_words.extend(self.word_tokenize(segement))
                w_segement = self.detokenize(w_segement_words)
                if HMCR >= np.random.random():
                    if PAR >= np.random.random():
                        if args.use_LLM:
                            if len(w_segement) > 0: 
                                candidate = self.gpt_adjuster(w_segement)
                            else:
                                candidate = self.gpt_adjuster(self.result_candidate)
                        else:
                            phrase_lookup = self.get_phrase_lookup(w_segement, args)
                            candidate, _ = self.perform_edit(edit_opertions_small, w_segement, phrase_lookup, delete_tracker)
                        w_segement = candidate
                    deleted = {}
                    added = {}
                else:
                    deleted = {}
                    added = {}
                    phrase_lookup = self.get_phrase_lookup(w_segement, args)
                    if args.use_LLM:               
                        if len(w_segement) > 0: 
                            candidate, deleted, added = self.mutated(w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                            w_segement = candidate
                        else:
                            candidate, deleted, added = self.mutated(self.result_candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                            w_segement = candidate
                    else:
                        candidates, deleted, added = self.mutated(w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                        w_segement = candidates[0] # multipule edit operations can be implemented if necessary
                w_m.append(w_segement + ',')
        else:
            for j in range(ks):
                idx = np.random.randint(0,len(self.W_candidates))
                w = self.W_candidates[idx]
                phrases_pun = self.get_phrase_lookup_pun(w, args)
                w_phrases = list(phrases_pun.values())
                L = len(w_phrases)
                start = math.ceil(j/ks*L)
                end = math.ceil((j+1)/ks*L) - 1
                if(start!=end):
                    w_segement = w_phrases[start:end]
                else:
                    w_segement = [w_phrases[start]]
                w_segement_words = []
                for phrase in w_segement:   
                    w_segement_words = w_segement_words + self.word_tokenize(phrase)
                w_segement = self.detokenize(w_segement_words)
                if HMCR >= np.random.random():
                    if PAR >= np.random.random():
                        # try:
                        #     if args.use_LLM:
                        #         print("w_segement1: ", w_segement)
                        #         if len(w_segement) > 0: 
                        #             candidate = self.gpt_adjuster(w_segement)
                        #         else:
                        #             phrase_lookup = self.get_phrase_lookup(w_segement, args)
                        #             candidate, _ = self.perform_edit(edit_opertions_small, w_segement, phrase_lookup, delete_tracker)
                        #     else:
                        #         phrase_lookup = self.get_phrase_lookup(w_segement, args)
                        #         candidate, _ = self.perform_edit(edit_opertions_small, w_segement, phrase_lookup, delete_tracker)
                        #     w_segement = candidate
                        # except:
                        #     print('Error occurs (parser) and skip this mutation 1')
                        #     continue
                        if args.use_LLM:
                            if len(w_segement) > 0: 
                                candidate = self.gpt_adjuster(w_segement)
                            else:
                                candidate = self.gpt_adjuster(self.result_candidate)
                        else:
                            phrase_lookup = self.get_phrase_lookup(w_segement, args)
                            candidate, _ = self.perform_edit(edit_opertions_small, w_segement, phrase_lookup, delete_tracker)
                        w_segement = candidate
                    deleted = {}
                    added = {}
                    
                else:
                    deleted = {}
                    added = {}
                    # try:
                    #     if args.use_LLM:
                    #         print("w_segement: ", w_segement)
                    #         candidates, deleted, added = self.mutated(w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                    #         print("candidates: ", candidates)
                    #     else:
                    #         phrase_lookup = self.get_phrase_lookup(w_segement, args)
                    #         candidates, deleted, added = self.mutated(w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                    #     w_segement = candidates[0] # multipule edit operations can be implemented if necessary
                    # except:
                    #     print('Error occurs (parser) and skip this mutation 2')
                    #     continue
                    phrase_lookup = self.get_phrase_lookup(w_segement, args)
                    if args.use_LLM:               
                        if len(w_segement) > 0: 
                            candidate, deleted, added = self.mutated(w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                            w_segement = candidate
                        else:
                            candidate, deleted, added = self.mutated(self.result_candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                            w_segement = candidate
                    else:
                        candidates, deleted, added = self.mutated(w_segement, phrase_lookup, use_add, delete_tracker, edit_operations, args)
                        w_segement = candidates[0] # multipule edit operations can be implemented if necessary
                w_m.append(w_segement)
        for segement in w_m:
            w_m_words.extend(self.word_tokenize(segement))
        w_m = self.detokenize(w_m_words)
        return w_m, deleted, added


    def train(self, instruction, chosen_task_name, args):
        
        ks = args.ks
        HMCR = args.HMCR
        PAR = args.PAR
        edit_opertions_small = 'sub'
        N_H = args.N_H

        meta_path = os.path.join(args.meta_dir, args.meta_name)
        # meta_file = open(meta_path, 'w+')
        edit_operations = args.edits
        use_add = 'add' in edit_operations

        if 'sub' in edit_operations:
            self.if_sub(edit_operations)

        self.init_population(instruction, args)

        # meta_file.write("Original Candidate:\t "+ self.original_candidate + '\n')
        # meta_file.write("Original Score:\t "+ str(self.original_score) + '\n')
        # meta_file.write("\n")
        print("Original Candidate:\t ", self.original_candidate)
        print("Original Score:\t ", self.original_score)
        print("")
        wandb.log({"original_score": self.original_score})
        current_iteration = 0 
        delete_tracker = []

        if len(args.resume):
            print("Resuming the searching from checkpoints...")
            self.load(args.resume)
            current_iteration, delete_tracker = self.set_state()
            
        while current_iteration < self.maxiter:
            current_iteration += 1
            print("====================Current Iteration: ", current_iteration)
            #Base_candidate after battled in the tournament
            base_candidate = self.result_candidate
            base_score = self.result_score

            # meta_file.write("Base Candidate:\t "+ base_candidate + '\n')
            # meta_file.write("Base Score:\t "+ str(base_score) + '\n')
            print("Base Candidate: ", base_candidate)
            print("Base Score: ", base_score)
            
            wandb.log({"step": current_iteration, "base_score": base_score})
            
            deleted_candidate = {}
            added_candidate = {}
            self.W_candidates_m = []
            self.W_scores_m = []
            for c in range(args.num_candidates):
                
                w_m, deleted, added = self.generate_candidate(ks, HMCR, PAR, edit_opertions_small, use_add, delete_tracker, edit_operations, args, current_iteration)
                print("Candidate: ", w_m)
                w_m_score = self.score(w_m, c+1, args=args)
                print("Candidate Score: ", w_m_score)
                self.W_candidates_m.append(w_m)
                self.W_scores_m.append(w_m_score)
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
                
            update_best_or_not = self.update_result(self.W_candidates_m, self.W_scores_m)
            if args.task_type == "text2image":
                if update_best_or_not:
                    best_idx = np.argmax(self.W_scores_m)
                    self.update_best_picture(best_idx+1, args)

            if self.patience_counter > args.patience:
                print('Ran out of patience')
                # meta_file.write('Ran out of patience \n')
                break
            
            self.W_candidates = self.W_candidates + self.W_candidates_m
            self.W_scores = self.W_scores + self.W_scores_m
            
            top_N_H_idx_list = heapq.nlargest(N_H, range(len(self.W_scores)), self.W_scores.__getitem__)
            W_candidates_top_N_H = [self.W_candidates[i] for i in top_N_H_idx_list]
            print("Top N Candidates of Current Iteration: ", W_candidates_top_N_H)
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
            
            if args.task_type == "text2text":
                self.result_candidate = self.detokenize(self.word_tokenize(self.result_candidate))
            elif args.task_type == "text2image":
                self.result_candidate = str(self.result_candidate)
                print("Result Candidate: ", self.result_candidate)
                print("Result Score: ", self.result_score)

            if current_iteration % args.checkpoint_freq == 0:
                self.get_state(current_iteration, delete_tracker)
                ckpt_dir = Path(args.output_dir) / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                filename = "task{}_step{}.pickle".format(args.task_idx, current_iteration-1)
                ckpt_path = ckpt_dir / filename
                self.save(ckpt_path)

            if args.task_type == "text2text":  
                if args.backbone == "gpt3":
                    count = gpt3.complete_gpt3.count
            
                if args.backbone == "gpt2":
                    count = gpt2.complete_gpt2.count
                    
                if count >= args.budget:
                    print('Ran out of budget')
                    break

            # if self.patience_counter > args.patience:
            #     print('Ran out of patience')
            #     meta_file.write('Ran out of patience \n')
            #     break
            # elif count >= args.budget:
            #     print('Ran out of budget')
            #     break
            # else: 
            #     continue

        wandb.log({"result_score": self.result_score})
        print('Final_Result Candidate: ', self.result_candidate)
        print('Final_Result Score: ', self.result_score)
        if args.task_type == "text2text":
            if args.backbone == "gpt3":
                count = gpt3.complete_gpt3.count
            
            if args.backbone == "gpt2":
                count = gpt2.complete_gpt2.count
                
            print('APICalls for search:\t', count)

            wandb.log({"apicalls_search": count})

            # meta_file.write('\n')

            searched_score = self.test(self.result_candidate, args)

            # meta_file.write('Testing .... \n')
            if args.print_orig:
                print('Task:\t', chosen_task_name)
                print('Original Instruction:\t', self.original_candidate)
                orig_score = self.score(self.original_candidate, 'test', args=args)
                print('Original Accuracy:\t', str(orig_score))
                # meta_file.write('Original Accuracy:\t'+ str(orig_score)+ '\n')

            if self.result_candidate == self.original_candidate: 
                print('No viable candidate found!')
                # meta_file.write('No viable candidate found!\n')
                print('APICalls:\t', count)
                # meta_file.write('APICalls:\t'+ str(count) + '\n')
                wandb.log({"Original Accuracy": orig_score})
                exit()

            wandb.log({"searched_accuracy": searched_score})
            wandb.log({"apicalls_total": count})

            print('Accuracy after search:\t', str(searched_score))
            print('Instruction after search:\t', self.result_candidate)
            # meta_file.write('Instruction after search:\t'+ self.result_candidate+ '\n')
            # meta_file.write('Accuracy after search:\t'+ str(searched_score)+ '\n')
            print('APICalls:\t', count)
            # meta_file.write('APICalls:\t'+ str(count) + '\n')

        wandb.save(meta_path)

    def test(self, instruction, args):

        print('\nTesting .... ')

        searched_score = self.score(instruction, 'test', write=args.write_preds, args=args)
        
        return searched_score




        


