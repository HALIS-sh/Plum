# export CUDA_VISIBLE_DEVICES=0
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
    # for ((ga_seed=start_seed; ga_seed<=end_seed; ga_seed++));
    # do
    pic_gen_seed=2
    ga_seed=2
    prefix="result/prompt-${line_number}_${prompt_hash}/ga_search_seed-${ga_seed}_picseed--${pic_gen_seed}"

    # 检查文件夹是否存在
    if [ ! -d "$prefix" ]; then
        # 如果文件夹不存在，则创建它
        mkdir "result/prompt-${line_number}_${prompt_hash}"
        mkdir "$prefix"
        mkdir "${prefix}/logs"
        mkdir "${prefix}/pics"
        echo "创建文件夹 $prefix"

        python pic_score_main.py \
            --train-seed ${ga_seed}   \
            --pic_gen_seed ${pic_gen_seed}  \
            --num-compose 1   \
            --num-candidates 10   \
            --num-iter 10  \
            --patience 7 \
            --pics_number 2 \
            --tournament-selection 5 \
            --task_type "text2image"  \
            --meta-dir "${prefix}/logs"\
            --meta-pic-dir "${prefix}/pics" \
            --meta-name "${prompt_hash}_ga_search_seed-${ga_seed}_picseed--${pic_gen_seed}.txt"   \
            --level "word" \
            --use_LLM 0  \
            --algorithm "ga" \
            --original_candidate "$content"| tee "${prefix}/logs/ga_search.log"
    else
        echo "文件夹 $prefix 已经存在"
    fi
    
done < "$file_path"