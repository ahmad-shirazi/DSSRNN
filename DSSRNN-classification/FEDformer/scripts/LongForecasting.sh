
module load python/3.9-2022.05 cuda/11.0.3
export CUDA_LAUNCH_BLOCKING=1

if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi


# ETTm1
python -u run.py \
  --is_training 1 \
  --data_path ds44.csv \
  --task_id ETTm1 \
  --model FEDformer \
  --data custom \
  --root_path ../dataset/ \
  --features MS \
  --seq_len 96 \
  --target zone_044_co2 \
  --label_len 48 \
  --pred_len 200 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --d_model 512 \
  --itr 1  >../logs/LongForecasting/FEDformer_ETTm1_$pred_len.log