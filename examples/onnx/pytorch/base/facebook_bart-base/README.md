# Example: Export [facebook/bart-base](https://huggingface.co/facebook/bart-base) to ONNX

## Environments

```
python==3.8.12
pytorch==1.10.0
transformers==4.12.5
onnxruntime-gpu==1.9.0
```

## Export to ONNX

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

