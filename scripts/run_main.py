import sys
sys.path.insert(0, "/content/jiant")

import argparse
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os

parser = argparse.ArgumentParser('Jiant parser')

parser.add_argument('--eval', action='store_true', help='Eval only run.')
parser.add_argument('--model', type=str, required=True, help='Hugging Face model string to use')
parser.add_argument('--task', type=str, required=True, help='The task name.')
parser.add_argument('--run-name', type=str, default='', help='The run name. Determines where checkpoints are stored')
parser.add_argument('--checkpoint-dir', type=str, default='/checkpoint/timdettmers/jiant/', help='The run name. Determines where checkpoints are stored')
parser.add_argument('--base-dir', type=str, default='/private/home/timdettmers/data/jiant_data', help='The run name. Determines where checkpoints are stored')
parser.add_argument('--no-fp16', action='store_true', help='Do not use fp16 training.')

args = parser.parse_args()

print(args)

export_model.export_model(
    hf_pretrained_model_name_or_path=args.model,
    output_base_path=f"{args.checkpoint_dir}/models/{args.model}",
)
tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
    #task_config_path=f"{args.checkpoint_dir}/runs/{args.task}/{args.model}/{args.run_name}/config.json",
    task_config_path=f"{args.base_dir}/tasks/configs/{args.task}_config.json",
    hf_pretrained_model_name_or_path=args.model,
    output_dir=f"{args.base_dir}/cache/{args.task}",
    phases=["train", "val"],
))

jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
    task_config_base_path=f"{args.base_dir}/tasks/configs",
    task_cache_base_path=f"{args.base_dir}/cache",
    train_task_name_list=[args.task],
    val_task_name_list=[args.task],
    train_batch_size=8,
    eval_batch_size=16,
    epochs=5,
    num_gpus=1,
).create_config()
os.makedirs(f"{args.base_dir}/run_configs/", exist_ok=True)
py_io.write_json(jiant_run_config, f"{args.base_dir}/run_configs/{args.task}_run_config.json")
display.show_json(jiant_run_config)

if not args.eval:
    model_path = f"{args.checkpoint_dir}/models/{args.model}/model/model.p"
else:
    model_path = f"{args.checkpoint_dir}/runs/{args.task}/{args.model}/{args.run_name}/best_model.p"

run_args = main_runscript.RunConfiguration(
    jiant_task_container_config_path=f"{args.base_dir}/run_configs/{args.task}_run_config.json",
    output_dir=f"{args.checkpoint_dir}/runs/{args.task}/{args.model}/{args.run_name}",
    hf_pretrained_model_name_or_path=args.model,
    model_path=model_path,
    #model_config_path=f"{args.checkpoint_dir}/runs/{args.task}/{args.model}/{args.run_name}/config.json",
    model_config_path=f"{args.base_dir}/models/{args.model}/model/config.json",
    learning_rate=1e-5,
    eval_every_steps=500,
    do_train=not args.eval,
    do_val=True,
    do_save=not args.eval,
    do_save_best=not args.eval,
    force_overwrite=not args.eval,
    fp16=not args.no_fp16,
    fp16_opt_level=('O1' if args.no_fp16 else 'O2'),
    model_load_mode = 'from_transformers' if not args.eval else 'all',
)
main_runscript.run_loop(run_args)
