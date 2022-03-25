from jiant.proj.main import runscript as run
import jiant.scripts.download_data.runscript as downloader

EXP_DIR = '/private/home/timdettmers/data/jiant_data/'

# Download the Data
downloader.download_data(["mrpc"], f"{EXP_DIR}/tasks")

# Set up the arguments for the Simple API
args = run.RunConfiguration(
   data_dir=f"{EXP_DIR}/tasks",
   hf_pretrained_model_name_or_path="roberta-base",
   tasks="mrpc",
   train_batch_size=16,
   num_train_epochs=3,
   keep_checkpoint_when_done=True
)

# Run!
run.run_simple(args)
