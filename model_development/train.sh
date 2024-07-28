#!/bin/bash
#SBATCH --qos=marasovic-gpulong-np
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --mem=245GB
#SBATCH --mail-user=u1420010@utah.edu
#SBATCH --mail-type=FAIL,REQUEUE,END
#SBATCH --requeue
#SBATCH -o report_train_f1_4.txt

echo "SLURM Nodename: $SLURMD_NODENAME"

source /uufs/chpc.utah.edu/common/home/u1420010/miniconda3/etc/profile.d/conda.sh
conda activate contract


#wandb disabled 
export TRANSFORMER_CACHE="/scratch/general/vast/u1420010/conda__cache"


GPU=1
# export CUDA_VISIBLE_DEVICES=$GPU


File_Src="preprocess/T5_ready_"
TASK_NAME="scifact-open"
TASK_NAME="contract-nli"

CHECKPOINT="/scratch/general/vast/u1420010/contract-nli/2/checkpoint-200"
# --resume_from_checkpoint $CHECKPOINT \

BATCH=1
EPOCH=8
LR=0.00001
for MODEL in google/flan-t5-xl
do
echo "---> RUNNING FOR "$MODEL;
time python3 generate_f1.py \
  --model_name_or_path $MODEL \
  --text_column "input" \
  --answer_column "choice" \
  --task_name $TASK_NAME \
  --per_device_eval_batch_size $BATCH \
  --per_device_train_batch_size $BATCH \
  --do_train \
  --do_eval \
  --do_predict \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --truncation_side "right" \
  --train_file $File_Src'train.json' \
  --validation_file $File_Src'dev.json' \
  --test_file $File_Src'test.json' \
  --max_source_length 4200 \
  --eval_steps 100 \
  --save_steps 100 \
  --logging_steps 100 \
  --save_strategy steps \
  --evaluation_strategy steps \
  --load_best_model_at_end \
  --gradient_checkpointing \
  --gradient_accumulation_steps 64 \
  --predict_with_generate \
  --overwrite_output_dir \
  --output_dir /scratch/general/vast/u1420010/$TASK_NAME/4
done

