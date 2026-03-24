OPENAI_API_KEY="-"
python ../tcwm_benchmark.py \
  --model_type api \
  --api_url http://localhost:8010/v1 \
  --model_name test \
  --api_key $OPENAI_API_KEY \
  --config_file ../configs/config_example.json \
  --output_dir ../results\
  --resume \
  # --skip_think   # 若模型不支持 CoT，可加此参数跳过 CoT 相关维度
