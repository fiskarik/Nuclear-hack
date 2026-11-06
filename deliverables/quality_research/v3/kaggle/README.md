# Prepared Kaggle GPU experiment

No remote submission has occurred. Keep the dataset and notebook private.
Use the existing cloud KAGGLE_API_TOKEN binding and HTTPS proxy. Never paste tokens into code or chat.
Run the following commands from this job directory.

1. Verify authentication: `kaggle kernels list --mine --page-size 1`.
2. Check `fiskarik/nuclear-hack-doe-training-v3` is not an unrelated dataset. If unused, run
   `kaggle datasets create -p dataset` without `--public`.
3. Wait until the dataset is ready, then run
   `kaggle kernels push -p notebook --accelerator NvidiaTeslaT4 -t 21600`.
4. Inspect `kaggle kernels status fiskarik/nuclear-hack-adaptive-gpu`. After complete, download the output
   with `kaggle kernels output fiskarik/nuclear-hack-adaptive-gpu -p NEW_DIRECTORY` and inspect actual metrics.

Saved files and a successful upload are not evidence of a completed GPU training run.
