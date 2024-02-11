export CUDA_VISIBLE_DEVICES=3
start_seed=0
end_seed=6
for ((hs_llm_seed=start_seed; hs_llm_seed<=end_seed; hs_llm_seed++));
do
    prefix="hs_llm_search_seed-${hs_llm_seed}"

    # 检查文件夹是否存在
    if [ ! -d "$prefix" ]; then
        # 如果文件夹不存在，则创建它
        mkdir "$prefix"
        mkdir "${prefix}/logs"
        mkdir "${prefix}/pics"
        echo "创建文件夹 $prefix"

        python pic_score_main.py \
            --train-seed ${hs_llm_seed}   \
            --num-compose 1   \
            --num-candidates 10   \
            --num-iter 10  \
            --patience 7 \
            --pics_number 2 \
            --task_type "text2image"  \
            --meta-dir "${prefix}/logs"\
            --meta-pic-dir "${prefix}/pics" \
            --meta-name "hs_llm_search_seed-${hs_llm_seed}.txt"   \
            --level "word" \
            --use_LLM 1  \
            --algorithm "hs_llm" \
            --original_candidate "Illustrate the sensation of falling in love as if it were a series of weather patterns unfolding across an expansive landscape"| tee "${prefix}/logs/hs_llm_search.log"
    else
        echo "文件夹 $prefix 已经存在"
    fi
done