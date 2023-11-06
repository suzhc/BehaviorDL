#!/bin/bash

# for d_model in 64 128 256 512
# do
#   for n_heads in 2 4 8
#   do
#     for e_layers in 1 2
#     do
#         for d_ff in 64 128 256 512
#         do
#             python exp_huawei.py \
#             --model 'PatchTST' \
#             --d_model $d_model \
#             --n_heads $n_heads \
#             --e_layers $e_layers \
#             --d_ff $d_ff \
#             --dropout 0.1 \
#             --factor 5 \
#             --activation 'relu' \
#             --seq_len 1441 \
#             --enc_in 4 \
#             --num_class 2 \
#             --num_features 4 \
#             --hidden_dim 128 \
#             --output_dim 2
#         done
#     done
#   done
# done


# for hidden_dim in 64 128 256 512
# do
#     python exp_huawei.py \
#     --model 'DLinear' \
#     --label_flag 'emotion' \
#     --d_model 256 \
#     --n_heads 4 \
#     --e_layers 2 \
#     --d_ff 256 \
#     --dropout 0.1 \
#     --factor 5 \
#     --activation 'relu' \
#     --seq_len 1441 \
#     --enc_in 4 \
#     --num_class 2 \
#     --num_features 4 \
#     --hidden_dim $hidden_dim \
#     --output_dim 2
# done

python exp_huawei.py \
    --model 'DLinear' \
    --label_flag 'emotion' \
    --d_model 256 \
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
    --model 'PatchTST' \
    --d_model 512 \
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
