export CUDA_VISIBLE_DEVICES=4
start_seed=0
end_seed=6
for ((hc_seed_2=start_seed; hc_seed_2<=end_seed; hc_seed_2++));
do
    prefix="hc_search_seed-${hc_seed_2}_prompt-2"

    # 检查文件夹是否存在
    if [ ! -d "$prefix" ]; then
        # 如果文件夹不存在，则创建它
        mkdir "$prefix"
        mkdir "${prefix}/logs"
        mkdir "${prefix}/pics"
        echo "创建文件夹 $prefix"

        python pic_score_main.py \
            --train-seed ${hc_seed_2}   \
            --num-compose 1   \
            --num-candidates 10   \
            --num-iter 10  \
            --patience 7 \
            --pics_number 2 \
            --task_type "text2image"  \
            --meta-dir "${prefix}/logs"\
            --meta-pic-dir "${prefix}/pics" \
            --meta-name "hc_search_seed-${hc_seed_2}_prompt-2.txt"   \
            --level "phrase" \
            --use_LLM 0  \
            --algorithm "hc" \
            --original_candidate "Show the boundary between night and day."| tee "${prefix}/logs/hc_search.log"
    else
        echo "文件夹 $prefix 已经存在"
    fi
done