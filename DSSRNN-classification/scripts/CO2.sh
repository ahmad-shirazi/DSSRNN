qmodel_name="SSRNN"  # Replace with your actual model name
seq_len="96" 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ds44-class.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 6 \
  --des 'Exp' \
  --itr 1 --batch_size 16  >logs/$model_name'_'Co2_$seq_len'_'96.log