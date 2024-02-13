# Set arrays for sequence lengths, datasets
pred_lengths=(96 192 336 720)
datasets=("ds44.csv" "ds45.csv" "ds62.csv" "ds68.csv")

# Iterate over each combination of sequence length, dataset, and model
for pred_len in "${pred_lengths[@]}"; do
  for data_file in "${datasets[@]}"; do

      # Create a unique job name
      job_name="job_${pred_len}_${data_file%.csv}"

      # Submit the job
      sbatch --job-name=$job_name <<EOF



# module load python/3.9 cuda/11.0.3
module load python/3.9-2022.05 cuda/11.0.3
export CUDA_LAUNCH_BLOCKING=1

if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi


# co2
python -u run.py \
  --is_training 1 \
  --data_path $data_file \
  --task_id co2 \
  --model FEDformer \
  --data custom \
  --root_path ../dataset/ \
  --features MS \
  --seq_len 96 \
  --target targetco2 \
  --label_len 72 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1  >../logs/LongForecasting/finaljob_${pred_len}_${data_file%.csv}.log
EOF

    done
  done
done
