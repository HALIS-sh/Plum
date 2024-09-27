# export CUDA_VISIBLE_DEVICES=0
file_path="test_prompts.txt"
# Initialize line number variable
line_number=0
# Use while loop to read the text file line by line
while IFS= read -r line; do
    # Increment line number
    ((line_number++))
    # Process or store the read content here
    content="$line"
    # Debugging output for line number and read content
    echo "Line number: $line_number"
    echo "Read content: $content"
    prompt_hash=$(echo "${content}" | md5sum | awk '{ print $1 }')
    # start_seed=0
    # end_seed=3
    # for ((ga_seed=start_seed; ga_seed<=end_seed; ga_seed++));
    # do
    pic_gen_seed=2
    ga_seed=2
    prefix="result/prompt-${line_number}_${prompt_hash}/ga_search_seed-${ga_seed}_picseed-${pic_gen_seed}"

    # Check if folder exists
    if [ ! -d "$prefix" ]; then
        # If folder does not exist, create it
        mkdir -p "result/prompt-${line_number}_${prompt_hash}"
        mkdir -p "$prefix"
        mkdir -p "${prefix}/logs"
        mkdir -p "${prefix}/pics"
        echo "Created folder $prefix"

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
        echo "Folder $prefix already exists"
    fi
    
done < "$file_path"