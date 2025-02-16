task="mmvet,mmbench_en_dev,mmmu,mme,pope,seedbench,ai2d,realworldqa,mvbench,tempcompass,videomme,activitynetqa"
python -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava_hf \
    --model_args pretrained="Kangheng/Video-UTR-7b" \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix video_utr_$task \
    --output_path ./logs/