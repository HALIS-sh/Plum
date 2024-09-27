# Set CUDA_VISIBLE_DEVICES to 1
# export CUDA_VISIBLE_DEVICES=1
file_path="test_prompts.txt"
# Initialize line number variable
line_number=0
# Read the text file line by line using a while loop
while IFS= read -r line; do
    # Increment the line number
    ((line_number++))
    # Process or store the content read
    content="$line"
    # Output the line number and content read for debugging
    echo "Line Number: $line_number"
    echo "Content Read: $content"
    # Calculate the MD5 hash of the content
    prompt_hash=$(echo "${content}" | md5sum | awk '{ print $1 }')
    # HS commas search seed and picture generation seed
    pic_gen_seed=2
    hs_seed=2
    prefix="result/prompt-${line_number}_${prompt_hash}/hs_commas_search_seed-${hs_seed}_picseed-${pic_gen_seed}"
    # Check if the folder does not exist
    if [ ! -d "$prefix" ]; then
        # If the folder does not exist, create it
        mkdir -p "result/prompt-${line_number}_${prompt_hash}"
        mkdir -p "$prefix"
        mkdir -p "${prefix}/logs"
        mkdir -p "${prefix}/pics"
        echo "Created folder $prefix"
        # Run the Python script with specified parameters and save the output to a log file
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
        echo "Folder $prefix already exists"
    fi
done < "$file_path"
