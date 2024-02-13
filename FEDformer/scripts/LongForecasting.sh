

# module load python/3.9 cuda/11.0.3
# module load python/3.9-2022.05 cuda/11.0.3
module load python/3.9 cuda/11.8.0
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
  --data_path ds44.csv \
  --task_id co2 \
  --model FEDformer \
  --data custom \
  --root_path ../dataset/ \
  --features MS \
  --seq_len 96 \
  --target zone_044_co2 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 20 \
  --d_layers 10 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1  >../logs/LongForecasting/FEDformer_GPU_co2_$pred_len.log