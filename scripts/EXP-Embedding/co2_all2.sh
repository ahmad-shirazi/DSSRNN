
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Embedding" ]; then
    mkdir ./logs/Embedding
fi

for datapath in ds44.csv ds45.csv ds62.csv ds68.csv
do
for model_name in DLinear Linear NLinear SSRNN DSSRNN
do 
for pred_len in 96 192 336 720
do

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $datapath \
  --model_id co2_$pred_len'_'$datapath \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len $pred_len \
  --enc_in 6 \
  --des 'Exp' \
  --itr 5 --batch_size 16  >logs/LongForecasting/$datapath'_'$model_name'co2test'$pred_len.log

done
done
done