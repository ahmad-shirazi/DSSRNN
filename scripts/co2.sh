model_name="DSSRNN"  # Replace with your actual model name
seq_len="96" 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ds44.csv \
  --model_id co2_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 6 \
  --des 'Exp' \
  --itr 5 --batch_size 16  >logs/LongForecasting/$model_name'_MSss'CO2_$seq_len'_'96.log