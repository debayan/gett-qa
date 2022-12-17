Training:

CUDA_VISIBLE_DEVICES=1 python -u run_summarization_no_trainer.py  --model_name_or_path t5-base  --train_file data/train_kgembed_jsonlines_1.json     
--validation_file data/dev_kgembed_jsonlines_1.json     --source_prefix "summarize: "     --output_dir model1    --per_device_train_batch_size=16  
--per_device_eval_batch_size=100 --learning_rate=1e-4 --num_train_epochs=100   | tee logs/log1.txt

Inference:

CUDA_VISIBLE_DEVICES=1 python -u t5_infer_kgembed.py  --model_name_or_path model1/epoch_99/  --validation_file data/test_kgembed_jsonlines_1.json    
--source_prefix "summarize: "   --per_device_eval_batch_size=50 --num_beams=3 | tee logs/logeval1.txt


