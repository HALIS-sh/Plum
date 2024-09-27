export CUDA_VISIBLE_DEVICES=2
for seed in 0 6
do
for TASK in  0 1 2 3 4 5 6 7 
do
  python ./run_search_modified_GA_C_add.py \
    --mode "Instruction Only" \
    --task-idx ${TASK} \
    --train-seed ${seed} \
    --num-compose 1 \
    --num-candidates 2 \
    --num-offspring 5 \
    --num-iter 50 \
    --patience 14 \
    --write-preds \
    --meta-dir "./logs/"\
    --meta-name "GA_C_batchsize_20_all_edits_l_1_m_2_mutationprob_05_n_50@task1_agnostic_trainseed_0_seed_42_rho_7_offspring5.txt" \
    --print-orig \
    --agnostic \
    --api-idx 1 \
    --batch-size 1 \
    --budget 10000 \
    --project-name 'gpt2-xl-gac-natural-instruction-match-12-8' \
    --model-name "gpt2" 




    # --algorithm 'ga' \
    # --data-dir "./natural-instructions/tasks/" \
    # --mode "Instruction Only" \
    # --task-idx ${TASK} \
    # --train-seed 0 \
    # --data-seed 42 \
    # --num-compose 1 \
    # --num-candidates 10 \
    # --num-iter 50 \
    # --patience 7 \
    # --write-preds \
    # --meta-dir "./logs/" \
    # --meta-name 'gpt2-HC_batchsize_20_all_edits_l_1_m_10_n_50_k_5_budget_1000@task7agnostic_trainseed_0_dataseed_42_rho_7.txt' \
    # --print-orig \
    # --agnostic \
    # --key-id 0 \
    # --batch-size 20 \
    # --tournament-selection 5 \
    # --project-name 'gpt3-gam-budget-new' \
    # --checkpoint-freq 10 \
    # --backbone 'gpt3' \
    # --output-dir './output'   # dir to save cheskpoints

    # add the following argument to resume the searching from the chechpoint
    # --resume /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle" 

    # add the following arguments to test the performance of the loaded model
    # --model-dir /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle 
    # --eval-only
done
done