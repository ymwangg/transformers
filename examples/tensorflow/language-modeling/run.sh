python run_mlm.py \
--model_name_or_path bert-base-uncased \
--do_train \
--output_dir /tmp/output \
--overwrite_output_dir true \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--max_seq_length 512 \
--per_device_train_batch_size 8

# amp max 22
# non-amp max 12
