export CUDA_VISIBLE_DEVICES=0,1

python -m pdb run.py \
    --text "诗人带着不甘回忆道："你我年少相逢，都有凌云之志"。" \
    --llm "hf-qwen3-vl-8b" \
    --outdir "test_outputs"