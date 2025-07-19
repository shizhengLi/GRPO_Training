

<div align="center">

# Training on my own

[Original Github repo](https://github.com/hkust-nlp/simpleRL-reason)


</div>


>This repo contains a simple reinforcement learning recipe to improve models' reasoning abilities. It is simple because only rule-based reward and GSM8K/Math datasets are used. We have used this code to successfully train 10 diverse base models with limited data (8K examples), achieving surprisingly strong results -- the accuracy gains range from 10 to more than 20 absolute points. These models include Llama3 8B, Mistral 7B/24B, DeepSeekMath 7B, Qwen2.5 0.5B/1.5B/7B/14B/32B, and Qwen2.5-Math-7B. While we observe significant increase in both response length and accuracy, we note that different models exhibit distinct reasoning behaviors during training, and the increased response length does not necessarily correlate with emergence of certain cognitive behaviors such as self-verification. We share many findings and practices in our paper, and we release the code, model checkpoints, and analysis tools here.  --from original repo.




Goal: Modify codes for my setting to run smoothly: 2 L20 (48 GB) GPUs.
Action: Tweaking hyperparameter...



## Quick Start

### Installation (Works for me)

Our code is implemented based on [Verl](https://github.com/volcengine/verl). We provide basic environment setup for training as follows, which only support custom environment setup and [FSDP training](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html). 

```bash
pay attention: python 3.9 doesn't work for me.!!!!
#conda create -n verl python==3.9
conda create -n verl python==3.10
conda activate verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e . 
pip3 install vllm==0.6.3
pip3 install ray
pip install wandb IPython matplotlib
```



### Reproducing SimpleRL-Zoo


#### Dataset

As mentioned in [our paper](http://arxiv.org/abs/2503.18892), our data includes three difficulty levels: Easy (GSM8K and MATH lv.1), Medium (MATH lv.1-4), and Hard (MATH lv.3-5). We have processed the data into two formats: simpler prompts (abel) and complex prompts (qwen), ready to use:


just download the [dataset](https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data) directly. E.g,

```
wget https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5/train.parquet
wget https://huggingface.co/datasets/hkust-nlp/SimpleRL-Zoo-Data/resolve/main/simplelr_qwen_level3to5/test.parquet
```
See the other folders for the other splits.


#### Training


The minimum hardware requirement for training Qwen-2.5-0.5B is a single H/A100-80G GPU. To accelerate our experiments, we utilized 2x8 H100-80G GPUs to train 7B and 14B models for approximately 100 steps over 15 hours using 8K examples. For training the 32B models, we used 8x8 H100-80G GPUs, completing the training in 1.5 days with the same dataset.

The training process leverages GRPO with Ray and vLLM for acceleration. So firstly, you need to launch the ray cluster using the command below:
```bash
# launch the master node of ray 
# Original
#ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
# My setting
ray start --head --node-ip-address 0.0.0.0 --num-gpus 2 --port 6380
```

Output
```bash

Local node IP: 0.0.0.0

Ray runtime started.

Next steps
  To add another node to this Ray cluster, run
    ray start --address='0.0.0.0:6380'
To connect to this Ray cluster:
    import ray
    ray.init(_node_ip_address='0.0.0.0')
To terminate the Ray runtime, run
    ray stop
To view the status of the cluster, use
    ray status

```

The main script for training is train_grpo_math_tune_ray.sh. You need to specify the required environment variables in this script. Once configured, submit the training job from the master node.

Here are examples for different models:

* Qwen-2.5-0.5B
My test:

```bash
RAY_ADDRESS='http://127.0.0.1:8265' bash train_grpo_math_tune_ray.sh --model_name Qwen-2.5-0.5B --max_response_length 2048 --train_batch_size 32 --rollout_n 2 --kl_loss_coef 0.0001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.5 --rollout_tp 2 --save_freq 5 --vllm_max_batched_tokens 4096 --max_val_seq_len 512 --disable_val_gen True

```

Original:
* Qwen-2.5-7B (For models between 0.5B and 14B, we use `kl_loss_coef=1e-4`)
```bash
bash train_grpo_math_tune_ray.sh --model_name Qwen-2.5-7B --max_response_length 8192  --train_batch_size 1024 --rollout_n 8 --kl_loss_coef 0.0001 --entropy_coeffient 0.001 --rollout_gpu_memory_util 0.75 --rollout_tp 2 --save_freq 5  
```


Note: The run name will depends on the model name and specific hyper-parameters to identify the training job. For example, above command will generate a run name like `verl-grpo_Qwen-2.5-32B_max_response8192_batch1024_rollout8_klcoef0.001_entcoef0.001_simplelr_math_35`. You can find the run name in terminal output. 



## Untest:

For other models, use the same command, adjusting the `--model_name` argument accordingly. 

### Evaluate

We used [Qwen Math's codebase](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) for evaluation, but for fairness considerations, we completely prohibited solving problems by calling code. The `eval_math_nodes.sh` script provides the full pipeline for evaluation, results collection, and analysis. To use it, you'll need to specify a few environment variables within the script, and then run it as shown below:

Example: 
```bash
bash eval_math_nodes.sh \
    --run_name verl-grpo_Qwen-2.5-32B_max_response8192_batch1024_rollout8_klcoef0.001_entcoef0.001_simplelr_math_35   \
    --init_model Qwen-2.5-32B \
    --template qwen-boxed  \
    --tp_size 8 \
    --add_step_0 true  \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_tokens 16000 \
    --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math \
    --n_sampling 1 
```

After running the script, the evaluation results will be saved in `$RUN_NAME/eval_results`, with the metrics from our paper (e.g., clip ratio, average response length, etc.) saved in `$RUN_NAME/eval_results/eval_results.csv`.

### Visualization

To compare the model's responses across different training steps, we offer a visualization tool that displays the model's reasoning process across various steps and benchmarks using Gradio. You can run the following script to access this tool:

```bash
# install gradio and httpx
pip install gradio
pip install httpx==0.23.0

bash launch_gradio.sh \
    --data_dir SimpleRL-verl/checkpoints \
    --run_names verl-grpo_Qwen-2.5-32B_max_response8192_batch1024_rollout8_klcoef0.001_entcoef0.001_simplelr_math_35  \
    --temperature 1.0   # temperature for evaluation
```



## Citation

If you find our paper/blog or our code useful, we would appreciate it if you could cite our work:

Cite our paper:
```bibtex
@misc{zeng2025simplerlzooinvestigatingtamingzero,
      title={SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild}, 
      author={Weihao Zeng and Yuzhen Huang and Qian Liu and Wei Liu and Keqing He and Zejun Ma and Junxian He},
      year={2025},
      eprint={2503.18892},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.18892}, 
}
```


Cite our blog:
```bibtex
@misc{zeng2025simplerl,
      title={7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient},
      author={Weihao Zeng and Yuzhen Huang and Wei Liu and Keqing He and Qian Liu and Zejun Ma and Junxian He},
      year={2025},
      howpublished={\url{https://hkust-nlp.notion.site/simplerl-reason}},
      note={Notion Blog}
}
```




