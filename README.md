# LM training scripts
- This repo contains a pre-training script to (continue) train small, Gemma-sized LMs. 
- There is an implementation of Gemma-2 with multiple embedding heads in `modeling_gemma.py`.

## Büble-2B evaluation
1. Clone the GermanBench repo: https://github.com/bjoernpl/GermanBenchmark
2. Set up a virtual environment
3. `pip install -r requirements.txt`

You can then run the following commands to get the benchmark scores. You can fill in any model you want to compare:

python main.py \
--model hf-causal \
--model_args pretrained=flair/bueble-lm-2b,dtype=float16 \
--tasks "arc_challenge_de,truthful_qa_de,hellaswag_de" \
--num_fewshot 0 \
--device cuda:0
