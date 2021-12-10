# Examples of exporting BART to ONNX

## Environments

```
python==3.8.12
pytorch==1.10.0
transformers==4.12.5
onnxruntime-gpu==1.9.0
```

## Example: Export [facebook/bart-base](https://huggingface.co/facebook/bart-base) to ONNX

1. Clone [facebook/bart-base](https://huggingface.co/facebook/bart-base)

   ```
   $ cd $REPO_DIR
   $ git lfs install
   $ git clone https://huggingface.co/facebook/bart-base
   ```

   * [$REPO_ROOT/INSTALLING.md in git-lfs/git-lfs (github)](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md)

2. Run `transformers.onnx`

   ```
   $ python -m transformers.onnx \\
   --model $REPO_DIR \\
   $ONNX_PATH
   ```

3. :exclamation:**TODO** Verify `$ONNX_PATH` by using [./verify.py](./verify.py)

## Example: Export [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) for sequence classification to ONNX

1. Clone [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)

   ```
   $ cd $REPO_DIR
   $ git lfs install
   $ git clone https://huggingface.co/facebook/bart-large-mnli
   ```

2. Run `transformers.onnx`

   ```
   $ python -m transformers.onnx \\
   --model $REPO_DIR \\
   --feature sequence-classification \\
   $ONNX_PATH
   ```
   
   <details>
     <summary>Click to expand!</summary>
     ```
       Using framework PyTorch: 1.10.0
       Overriding 1 configuration item(s)
               - use_cache -> False
       /data/swook/miniconda3/envs/transformers-latest/lib/python3.8/site-packages/torch/onnx/utils.py:90: UserWarning: 'enable_onnx_checker' is deprecated and ignored. It will be removed in the next PyTorch release. To proceed despite ONNX checker failures, catch torch.onnx.ONNXCheckerError.
         warnings.warn("'enable_onnx_checker' is deprecated and ignored. It will be removed in "
       /data/swook/miniconda3/envs/transformers-latest/lib/python3.8/site-packages/torch/onnx/utils.py:103: UserWarning: `use_external_data_format' is deprecated and ignored. Will be removed in next PyTorch release. The code will work as it is False if models are not larger than 2GB, Otherwise set to False because of size limits imposed by Protocol Buffers.
         warnings.warn("`use_external_data_format' is deprecated and ignored. Will be removed in next "
       /data/swook/repos/huggingface/transformers-cloned/src/transformers/models/bart/modeling_bart.py:217: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
         if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
       /data/swook/repos/huggingface/transformers-cloned/src/transformers/models/bart/modeling_bart.py:223: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
         if attention_mask.size() != (bsz, 1, tgt_len, src_len):
       /data/swook/repos/huggingface/transformers-cloned/src/transformers/models/bart/modeling_bart.py:254: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
         if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
       /data/swook/repos/huggingface/transformers-cloned/src/transformers/models/bart/modeling_bart.py:888: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
         if input_shape[-1] > 1:
       /data/swook/repos/huggingface/transformers-cloned/src/transformers/models/bart/modeling_bart.py:1482: TracerWarning: Using len to get tensor shape might cause the trace to be incorrect. Recommended usage would be tensor.shape[0]. Passing a tensor of different shape might lead to errors or silently give incorrect results.
         if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
       Validating ONNX model...
       /data/swook/miniconda3/envs/transformers-latest/lib/python3.8/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:350: UserWarning: Deprecation warning. This ORT build has ['CUDAExecutionProvider', 'CPUExecutionProvider'] enabled. The next release (ORT 1.10) will require explicitly setting the providers parameter (as opposed to the current behavior of providers getting set/registered by default based on the build flags) when instantiating InferenceSession.For example, onnxruntime.InferenceSession(..., providers=["CUDAExecutionProvider"], ...)
         warnings.warn("Deprecation warning. This ORT build has {} enabled. ".format(available_providers) +
               -[✓] ONNX model outputs' name match reference model ({'encoder_last_hidden_state', 'logits'}
               - Validating ONNX Model output "logits":
                       -[✓] (2, 3) matches (2, 3)
                       -[✓] all values close (atol: 0.0001)
               - Validating ONNX Model output "encoder_last_hidden_state":
                       -[✓] (2, 8, 1024) matches (2, 8, 1024)
                       -[✓] all values close (atol: 0.0001)
       All good, model saved at: $ONNX_PATH/model.onnx
     ```
   </details>
   
3. :exclamation:**TODO** Verify `$ONNX_PATH` by using [./verify.py](./verify.py)

