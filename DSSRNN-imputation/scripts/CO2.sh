seq_len=50
model_name=SSRNN

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ds62p1.csv \
  --model_id co2p1_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len 1 \
  --enc_in 6 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/LongForecasting/$model_name'_'Co2_$seq_len'_'1-p1-impute.log