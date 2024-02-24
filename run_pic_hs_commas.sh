# export CUDA_VISIBLE_DEVICES=1
file_path="test_prompts.txt"
# 初始化行号变量
line_number=0
# 使用while循环按行读取文本文件
while IFS= read -r line; do
    # 递增行号
    ((line_number++))
     # 在此处对读取到的内容进行处理或存储到变量中
    content="$line"
    # 在此处输出行号和读取到的内容，用于调试
    echo "读取到的行号： $line_number"
    echo "读取到的内容： $content"
    prompt_hash=$(echo "${content}" | md5sum | awk '{ print $1 }')
    # start_seed=0
    # end_seed=3
    # for ((hs_seed=start_seed; hs_seed<=end_seed; hs_seed++));
    # do
    pic_gen_seed=2
    hs_seed=2
    prefix="result/prompt-${line_number}_${prompt_hash}/hs_commas_search_seed-${hs_seed}_picseed-${pic_gen_seed}"

    # 检查文件夹是否存在
    if [ ! -d "$prefix" ]; then
        # 如果文件夹不存在，则创建它
        mkdir "result/prompt-${line_number}_${prompt_hash}"
        mkdir "$prefix"
        mkdir "${prefix}/logs"
        mkdir "${prefix}/pics"
        echo "创建文件夹 $prefix"

        python pic_score_main.py \
            --train-seed ${hs_seed}   \
            --num-compose 1   \
            --num-candidates 10   \
            --num-iter 10  \
            --patience 7 \
            --pics_number 2 \
            --task_type "text2image"  \
            --meta-dir "${prefix}/logs"\
            --meta-pic-dir "${prefix}/pics" \
            --meta-name "/${prompt_hash}_hs_commas_search_seed-${hs_seed}_picseed--${pic_gen_seed}.txt"   \
            --level "word" \
            --use_commas_split 1 \
            --use_LLM 0  \
            --algorithm "hs" \
            --original_candidate "$content"| tee "${prefix}/logs/hs_commas_search.log"
    else
        echo "文件夹 $prefix 已经存在"
    fi
    
done < "$file_path"