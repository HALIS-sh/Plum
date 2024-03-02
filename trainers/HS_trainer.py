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
from abc import abstractmethod

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
    
    @abstractmethod
    def gpt_adjuster(self, sentence):
        pass


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
                # As the number of phrases of some prompt maybe less than 5, to guarantee the normal operation of the algorithm, we need to extend the phrases to 5.
                if L < 5:
                    # Because the prompt is split by commas, the last phrase may be empty, so we need to remove it.
                    if L == 1:
                        parts[0] = parts[0] + ','
                    # If the number of phrases is less than 5, we need to extend the phrases to at least 5.
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
        if args.task_type == "text2text":
            meta_file = open(meta_path, 'w+')
        edit_operations = args.edits
        use_add = 'add' in edit_operations

        if 'sub' in edit_operations:
            self.if_sub(edit_operations)

        self.init_population(instruction, args)

        if args.task_type == "text2text":
            meta_file.write("Original Candidate:\t "+ self.original_candidate + '\n')
            meta_file.write("Original Score:\t "+ str(self.original_score) + '\n')
            meta_file.write("\n")
        else:
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

            if args.task_type == "text2text":
                meta_file.write("Base Candidate:\t "+ base_candidate + '\n')
                meta_file.write("Base Score:\t "+ str(base_score) + '\n')
            else:
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
                if args.task_type == "text2text":
                    meta_file.write('Ran out of patience \n')
                else:
                    print('Ran out of patience')
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
            if args.task_type == "text2text":
                if self.patience_counter > args.patience:
                    print('Ran out of patience')
                    meta_file.write('Ran out of patience \n')
                    break
                elif count >= args.budget:
                    print('Ran out of budget')
                    break
                else: 
                    continue

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

            meta_file.write('\n')

            searched_score = self.test(self.result_candidate, args)

            meta_file.write('Testing .... \n')
            if args.print_orig:
                print('Task:\t', chosen_task_name)
                print('Original Instruction:\t', self.original_candidate)
                orig_score = self.score(self.original_candidate, 'test', args=args)
                print('Original Accuracy:\t', str(orig_score))
                meta_file.write('Original Accuracy:\t'+ str(orig_score)+ '\n')

            if self.result_candidate == self.original_candidate: 
                print('No viable candidate found!')
                meta_file.write('No viable candidate found!\n')
                print('APICalls:\t', count)
                meta_file.write('APICalls:\t'+ str(count) + '\n')
                wandb.log({"Original Accuracy": orig_score})
                exit()

            wandb.log({"searched_accuracy": searched_score})
            wandb.log({"apicalls_total": count})

            print('Accuracy after search:\t', str(searched_score))
            print('Instruction after search:\t', self.result_candidate)
            meta_file.write('Instruction after search:\t'+ self.result_candidate+ '\n')
            meta_file.write('Accuracy after search:\t'+ str(searched_score)+ '\n')
            print('APICalls:\t', count)
            meta_file.write('APICalls:\t'+ str(count) + '\n')

        wandb.save(meta_path)

    def test(self, instruction, args):

        print('\nTesting .... ')

        searched_score = self.score(instruction, 'test', write=args.write_preds, args=args)
        
        return searched_score




        


