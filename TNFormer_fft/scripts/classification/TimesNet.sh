export CUDA_VISIBLE_DEVICES=3

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 14 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10
