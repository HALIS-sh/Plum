import argparse
import json
from trainers import Pic_HC_trainer, Pic_GA_trainer, Pic_HS_trainer, Pic_HC_LLM_trainer, Pic_HS_LLM_trainer
import wandb

def main(args):

    num_compose = args.num_compose
    num_candidates = args.num_candidates
    train_seed = args.train_seed
    patience = args.patience
    num_steps = args.num_iter
    data_seed = args.data_seed
    task_type = args.task_type

    if args.algorithm == "hc":
        trainer = Pic_HC_trainer.Pic_HC_trainer(num_steps, patience, train_seed, data_seed, num_compose, num_candidates, backbone="", task_type=task_type)
    elif args.algorithm == "ga": 
        trainer = Pic_GA_trainer.Pic_GA_trainer(num_steps, patience, train_seed, data_seed, num_compose, num_candidates)
    elif args.algorithm == "hs":
        trainer = Pic_HS_trainer.Pic_HS_trainer(num_steps, patience, train_seed, data_seed, num_compose, num_candidates, backbone="", task_type=task_type)
    elif args.algorithm == "hc_llm":
        trainer = Pic_HC_LLM_trainer.Pic_HC_LLM_trainer(num_steps, patience, train_seed, data_seed, num_compose, num_candidates, backbone="", task_type=task_type)
    elif args.algorithm == "hs_llm":
        trainer = Pic_HS_LLM_trainer.Pic_HS_LLM_trainer(num_steps, patience, train_seed, data_seed, num_compose, num_candidates, backbone="", task_type=task_type)
    trainer.initialize_prompt_0(args)
    instruction = trainer.original_candidate
    trainer.train(instruction, chosen_task_name="pic_score", args = args)
    # trainer.test_gpt_rephraser(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="Instruction Only", help='Mode of instructions/prompts')
    parser.add_argument('--model-name', default="text-babbage-001", help='Name of used model')
    parser.add_argument('--num-shots', default=2, type=int, help='Number of examples in the prompt if applicable')
    parser.add_argument('--batch-size', default=4, type=int, help='Batch size')
    parser.add_argument('--task-idx', default=1, type=int, help='The index of the task based on the array in the code')
    parser.add_argument('--data-seed', default=42, type=int, help='Seed that changes score dataset by sampling examples')
    parser.add_argument('--train-seed', type=int, help='Seed that changes the sampling of edit operations (search seed)')
    parser.add_argument('--num-compose', default=1, type=int, help='Number of edits composed to get one candidate')
    parser.add_argument('--num-train', default=100, type=int, help='Number of examples in score set')
    parser.add_argument('--level', default="phrase", help='Level at which edit operations occur')
    parser.add_argument('--simulated-anneal', action='store_true', default=False, help='Runs simulated anneal if candidate scores <= base score')
    parser.add_argument('--agnostic', action='store_true', default=False, help='Uses template task-agnostic instruction')
    parser.add_argument('--print-orig', action='store_true', default=False, help='Print original instruction and evaluate its performance')
    parser.add_argument('--write-preds', action='store_true', default=False, help='Store predictions in a .json file')
    parser.add_argument('--data-dir', default='./natural-instructions-2.6/tasks/', help='Path to the dataset')
    parser.add_argument('--meta-dir', default='logs/', help='Path to store metadata of search')
    parser.add_argument('--meta-pic-dir', default='pics/', help='Path to store metadata of search')
    parser.add_argument('--meta-name', default='search.txt', help='Path to the file that stores metadata of search')
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument('--patience', default=2, type=int, help='The max patience P (counter)')
    parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates in each iteration (m)')
    parser.add_argument('--num-iter', default=10, type=int, help='Max number of search iterations')
    parser.add_argument('--key-id', default=0, type=int, help='Use if you have access to multiple Open AI keys')
    parser.add_argument('--edits', nargs="+", default=['del', 'swap', 'sub', 'add'], help='Space of edit ops to be considered')
    parser.add_argument('--tournament-selection', default=3, type=int, help='Number of tournament selections')
    parser.add_argument('--project-name', default='evolutional-prompt', help='Name of the wandb project')
    parser.add_argument('--num-samples', default=100, type=int, help='size of score set, default is 100')
    parser.add_argument('--classification-task-ids', default=['019', '021', '022', '050', '069', '137', '139', '195'], type=list, help='classification tasks')
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Checkpoint every N steps.")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument('--backbone', default="gpt3", help='backbone model')
    parser.add_argument('--algorithm', default="ga", help='Searching Algorithms')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    parser.add_argument('--budget', default=1000, type=int, help='number of the budget of api calls for searching')
    parser.add_argument('--api-idx', type=int, default=0)
    parser.add_argument('--pics_number', default=2, type=int, help='number of pictures used to calculate the score')
    parser.add_argument('--task_type', default="text2image", help='task type')
    parser.add_argument('--use_LLM', type=int, default=0, help='use LLM to generate prompt')
    parser.add_argument('--original_candidate', type=str, help='original candidate')
    parser.add_argument('--ks', type=int, default=2, help='key index')
    parser.add_argument('--HMCR', type=float, default=0.4, help='Harmony Memory Consideration Rate')
    parser.add_argument('--PAR', type=float, default=0.5, help='Pitch Adjustment Rate')
    parser.add_argument('--N_H', type=int, default=10, help='Number of Harmony')
    args = parser.parse_args()
    
    # # Initialize wandb
    wandb.login(key="88bd7de7f3b87f2354afdc42c30ae6cacba5ff0b")
    wandb.init(project=args.project_name, name=args.meta_name)
    wandb.config.update(args)

    main(args)