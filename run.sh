python exp_huawei.py \
    --model 'PatchTST' \
    --d_model 64 \
    --n_heads 4 \
    --e_layers 2 \
    --d_ff 256 \
    --dropout 0.1 \
    --factor 5 \
    --activation 'relu' \
    --seq_len 1441 \
    --enc_in 4 \
    --num_class 2 \
    --num_features 4 \
    --hidden_dim 128 \
    --output_dim 2

python exp_huawei.py \
    --model 'DLinear' \
    --d_model 64 \
    --n_heads 4 \
    --e_layers 2 \
    --d_ff 256 \
    --dropout 0.1 \
    --factor 5 \
    --activation 'relu' \
    --seq_len 1441 \
    --enc_in 4 \
    --num_class 2 \
    --num_features 4 \
    --hidden_dim 128 \
    --output_dim 2

