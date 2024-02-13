if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Embedding" ]; then
    mkdir ./logs/Embedding
fi

for datapath in ds44.csv ds45.csv ds62.csv ds68.csv
do
for model_name in Autoformer Informer Transformer
do 
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path $datapath \
    --model_id exchange_96_$pred_len \
    --model $model_name \
    --data custom \
    --features MS \
    --seq_len 96 \
    --use_gpu True \
    --label_len 48 \
    --use_gpu True \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 6 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 5 >logs/Embedding/$datapath'_'$model_name'co2test'$pred_len.log

done
done
done