# minGPT Lightning Benchmark

![mingpt](mingpt.jpg)

Modified [Andrej's](https://github.com/karpathy/minGPT) and [William's](https://github.com/williamFalcon/minGPT) awesome code to create a simple benchmarking script.

### Usage

```
pip install -r requirements.txt
```

### Results

Results were collected on an 8 GPU A100 server.

#### Maximum DeepSpeed

Largest model I could fit onto 8 A100 GPUs with 990GB of CPU RAM

##### DeepSpeed ZeRO Stage 3 Offload

```
~20B
python benchmark.py --n_layer 21 --n_head 16 --n_embd 8192 --gpus 8 --plugins deepspeed --precision 16 --limit_train_batches 120 --batch_size 1

Average Epoch time: 45.65 seconds
Average Peak memory 36086.14MiB
```

#### Smaller Model Comparison, DDP vs DeepSpeed

We collected results using a model size that fit training with DDP (roughly 1.6B parameters). 

This benchmark simulates the improvement in memory when training larger models, which is useful for users that do not have access to high memory GPUs.

A technical note: when using DeepSpeed, I noticed that for the first 20 batches, the optimizer step were skipped as infs were detected.

Command:
```bash
1.6B
python benchmark.py --n_layer 14 --n_head 16 --n_embd 3072 --gpus 8 --precision 16 --limit_train_batches 128
```

##### DDP
```
python benchmark.py --n_layer 14 --n_head 16 --n_embd 3072 --gpus 8 --precision 16 --limit_train_batches 128
TODO
```

##### DeepSpeed ZeRO Stage 3
```
Average Epoch time: 42.89 seconds
Average Peak memory 8813.29MiB
```

##### DeepSpeed ZeRO Stage 3 Offload
```
Average Epoch time: 151.12 seconds
Average Peak memory 5804.81MiB
```

#### Instantiating DeepSpeed

When modifying defaults, we have to specify the ``DeepSpeedPlugin`` as input, so I made the modification as such to the benchmark script and adjusted parameters when necessary:

```python
from pytorch_lightning.plugins import DeepSpeedPlugin

...
trainer = Trainer.from_argparse_args(
    args,
    max_epochs=1,
    gradient_clip_val=1.0,
    plugins=[DeepSpeedPlugin(zero_optimization=False)], # Pass in my own custom deepspeed plugin to turn off ZeRO-Offload
    callbacks=[lr_decay, CUDACallback()],
)
```

### References

Code:

- [openai/gpt-2](https://github.com/openai/gpt-2) has the model but not the training code, and in TensorFlow
- [openai/image-gpt](https://github.com/openai/image-gpt) has some more modern gpt-3 like modification in its code, good reference as well
- huggingface/transformers has a [language-modeling example](https://github.com/huggingface/transformers/tree/master/examples/language-modeling). It is full-featured but as a result also somewhat challenging to trace. E.g. some large functions have as much as 90% unused code behind various branching statements that is unused in the default setting of simple language modeling.
- [Teddy Koker/image-gpu in PyTorch Lightning](https://github.com/teddykoker/image-gpt)

Papers + some implementation notes:

#### Improving Language Understanding by Generative Pre-Training (GPT-1)

- Our model largely follows the original transformer work
- We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states.
- Adam max learning rate of 2.5e-4. (later GPT-3 for this model size uses 6e-4)
- LR decay: increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule
- We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.
- Since layernorm is used extensively throughout the model, a simple weight initialization of N(0, 0.02) was sufficient
- bytepair encoding (BPE) vocabulary with 40,000 merges
- residual, embedding, and attention dropouts with a rate of 0.1 for regularization.
- modified version of L2 regularization proposed in (37), with w = 0.01 on all non bias or gain weights
- For the activation function, we used the Gaussian Error Linear Unit (GELU).
- We used learned position embeddings instead of the sinusoidal version proposed in the original work
- For finetuning: We add dropout to the classifier with a rate of 0.1. learning rate of 6.25e-5 and a batchsize of 32. 3 epochs. We use a linear learning rate decay schedule with warmup over 0.2% of training. λ was set to 0.5.
- GPT-1 model is 12 layers and d_model 768, ~117M params

#### Language Models are Unsupervised Multitask Learners (GPT-2)

- LayerNorm was moved to the input of each sub-block, similar to a pre-activation residual network
- an additional layer normalization was added after the final self-attention block.
- modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers. (weird because in their released code i can only find a simple use of the old 0.02... in their release of image-gpt I found it used for c_proj, and even then only for attn, not for mlp. huh. https://github.com/openai/image-gpt/blob/master/src/model.py)
- the vocabulary is expanded to 50,257
- increase the context size from 512 to 1024 tokens
- larger batchsize of 512 is used
- GPT-2 used 48 layers and d_model 1600 (vs. original 12 layers and d_model 768). ~1.542B params

#### Language Models are Few-Shot Learners (GPT-3)

- GPT-3: 96 layers, 96 heads, with d_model of 12,288 (175B parameters).
- GPT-1-like: 12 layers, 12 heads, d_model 768 (125M)
- We use the same model and architecture as GPT-2, including the modified initialization, pre-normalization, and reversible tokenization described therein
- we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer
- we always have the feedforward layer four times the size of the bottleneck layer, dff = 4 ∗ dmodel
- all models use a context window of nctx = 2048 tokens.
- Adam with β1 = 0.9, β2 = 0.95, and eps = 10−8
- All models use weight decay of 0.1 to provide a small amount of regularization. (NOTE: GPT-1 used 0.01 I believe, see above)
- clip the global norm of the gradient at 1.0
- Linear LR warmup over the first 375 million tokens. Then use cosine decay for learning rate down to 10% of its value, over 260 billion tokens.
- gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size.
- full 2048-sized time context window is always used, with a special END OF DOCUMENT token delimiter

### License

MIT
