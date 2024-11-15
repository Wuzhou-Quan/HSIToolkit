import gsettings, torch, os, mmengine, torch.optim as optim, torchvision.transforms as v2, datetime, click, torch.utils.data, pathlib
from mmengine.runner import Runner

try:
    import clearml
    from mmengine.visualization import ClearMLVisBackend

    is_clearml_installed = True
except ImportError:
    is_clearml_installed = False
from inad_hsi_toolkit.metrics.mmengine_metrics import PixelRestoration
from mmengine.config import Config
from typing import Union
from mmengine.hooks import CheckpointHook

project_name = "superconv"

# Examples (HSDT_s Stage1 Gaussian50)
# CUDA_VISIBLE_DEVICES=0 python train.py --model_arch_name hsdt_s --train_dataset_name icvl_half_part1 --train_noise_type gaussian50 --val_dataset_name icvl_test --val_noise_type gaussian50 --optim_sche_strategy hsdtstage1 --batch_size=4 --max_epoches 30 --save_cp_interval 10 --lr 1e-3

# Examples (HSDT_s Stage2 Gaussian70Blind)
# CUDA_VISIBLE_DEVICES=0 python train.py --model_arch_name hsdt_s --train_dataset_name icvl_half_part1 --train_noise_type gaussian70blind --val_dataset_name icvl_test --val_noise_type gaussian70blind --optim_sche_strategy hsdtstage2 --batch_size=4 --max_epoches 50 --save_cp_interval 10 --lr 1e-3 --ckpt_path 'exps/hsdt_s_icvl_half_part1_gaussian50/best_PixelRestoration_PSNR_epoch_18.pth'


@click.command()
@click.option(
    "--model_arch_name",
    prompt="Enter the model architecture name from the deployments folder",
    type=str,
    required=True,
    help="Specify the architecture name of the model located in the deployments folder.",
)
@click.option(
    "--train_dataset_name",
    prompt="Enter the training dataset name from the deployments folder",
    type=str,
    required=True,
    help="Specify the name of the training dataset located in the deployments folder.",
)
@click.option(
    "--val_dataset_name",
    prompt="Enter the validation dataset name from the deployments folder",
    type=str,
    required=True,
    help="Specify the name of the validation dataset located in the deployments folder.",
)
@click.option(
    "--train_noise_type",
    prompt="Enter the noise mode for training (default is 'None')",
    type=str,
    default="None",
    help="Specify the noise type used during training.",
)
@click.option(
    "--val_noise_type",
    prompt="Enter the noise mode for validation (default is 'None')",
    type=str,
    default="None",
    help="Specify the noise type used during validation.",
)
@click.option(
    "--optim_sche_strategy",
    prompt="Enter the optimization schedule strategy (default is 'default')",
    type=str,
    default="default",
    help="Specify the optimization schedule strategy to use (e.g., 'default', 'step', 'cosine').",
)
@click.option("--batch_size", prompt="Enter the batch size (default is 4)", type=int, default=4, help="Specify the batch size for training.")
@click.option("--max_epoches", prompt="Enter the maximum number of epochs (default is 100)", type=int, default=100, help="Specify the maximum number of training epochs.")
@click.option("--ckpt_path", prompt="Enter the checkpoint file path (optional)", type=str, default=None, help="Specify the path to the checkpoint file if you want to resume from a checkpoint.")
@click.option(
    "--with_aug",
    prompt="Do you want to apply data augmentation during training? (default is True)",
    type=bool,
    default=True,
    help="Indicate whether to use data augmentation during training (True/False).",
)
@click.option("--lr", prompt="Enter the learning rate (default is 1e-3)", type=float, default=1e-3, help="Specify the learning rate for the optimizer.")
@click.option(
    "--save_cp_interval", prompt="Enter the checkpoint saving interval (default is 100)", type=int, default=100, help="Specify the interval (in epochs) at which to save the model checkpoints."
)
@click.option("--val_interval", prompt="Enter the validation interval (default is 1)", type=int, default=1, help="Specify the interval (in epochs) at which to run validation.")
def start_one_training_process(
    model_arch_name: str,
    train_dataset_name: str,
    val_dataset_name: str,
    train_noise_type: str,
    val_noise_type: str,
    batch_size: int,
    max_epoches: int,
    ckpt_path: str,
    with_aug: bool,
    lr: float,
    save_cp_interval: int,
    val_interval: int,
    optim_sche_strategy: str,
):
    # Dataset
    train_dataset_deployment_file_path = gsettings.deployment_path / "datasets" / f"{train_dataset_name}.py"
    val_dataset_deployment_file_path = gsettings.deployment_path / "datasets" / f"{val_dataset_name}.py"
    train_dataset_cfg = Config.fromfile(
        os.fspath(train_dataset_deployment_file_path),
        lazy_import=False,
    )
    val_dataset_cfg = Config.fromfile(
        os.fspath(val_dataset_deployment_file_path),
        lazy_import=False,
    )
    if with_aug:
        train_dataset_cfg.dataset["augments"] = v2.Compose(
            [
                v2.ToTensor(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ConvertImageDtype(torch.float32),
            ]
        )
    train_noise_maker_file_path = gsettings.deployment_path / "noise_makers" / f"{train_noise_type}.py" if train_noise_type.upper() != "NONE" else None
    if train_noise_maker_file_path:
        train_dataset_cfg.dataset["noise_maker"] = Config.fromfile(
            os.fspath(train_noise_maker_file_path),
            lazy_import=False,
        ).noise_maker
    val_noise_maker_file_path = gsettings.deployment_path / "noise_makers" / f"{val_noise_type}.py" if val_noise_type.upper() != "NONE" else None
    if val_noise_maker_file_path:
        val_dataset_cfg.dataset["noise_maker"] = Config.fromfile(
            os.fspath(val_noise_maker_file_path),
            lazy_import=False,
        ).noise_maker
    train_dataloader = torch.utils.data.DataLoader(
        mmengine.DATASETS.build(train_dataset_cfg.dataset), batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False, prefetch_factor=2, persistent_workers=True, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        mmengine.DATASETS.build(val_dataset_cfg.dataset), batch_size=1, shuffle=True, num_workers=2, drop_last=False, prefetch_factor=2, persistent_workers=True, pin_memory=True
    )

    # Model
    model_deployment_file_path = gsettings.deployment_path / "models" / f"{model_arch_name}.py"
    model_cfg = Config.fromfile(
        os.fspath(model_deployment_file_path),
        lazy_import=False,
    )

    def replace_numbands(cfg: Union[dict, Config], numbands):
        for k, v in cfg.items():
            if isinstance(v, dict):
                replace_numbands(v, numbands)
            elif v == "%numbands%":
                cfg[k] = numbands

    replace_numbands(model_cfg, train_dataset_cfg.channels)
    model: torch.nn.Module = mmengine.MODELS.build(model_cfg.model)

    # Optimizer Settings
    optim_wrapper = dict(optimizer=dict(type=optim.AdamW, lr=lr))
    scheduler_file_path = gsettings.deployment_path / "optim_sche" / f"{optim_sche_strategy}.py"
    scheduler_cfg = Config.fromfile(
        os.fspath(scheduler_file_path),
        lazy_import=False,
    )
    param_scheduler = scheduler_cfg.param_scheduler

    # Checkpoints
    if ckpt_path is not None:
        ckpt_path = os.fspath(pathlib.Path(ckpt_path).absolute())
    assert ckpt_path is None or os.path.exists(ckpt_path), f"Checkpoint path {ckpt_path} not found"

    if is_clearml_installed:
        visualizer = dict(
            type="Visualizer",
            vis_backends=[
                dict(
                    type=ClearMLVisBackend,
                    init_kwargs=dict(
                        task_name=f"{model_arch_name} / {train_dataset_name}{'_noaug' if not with_aug else ''} / {train_noise_type}",
                        project_name=project_name,
                    ),
                )
            ],
        )
    else:
        visualizer = None
    runner = Runner(
        model=model,
        experiment_name=f"{model_arch_name}_{train_dataset_name}_{train_noise_type}{'_noaug' if not with_aug else ''}_{datetime.datetime.now().strftime(r'%y%m%d_%H%M%S')}",
        work_dir=os.fspath(gsettings.exps_workdir_path / f"{model_arch_name}_{train_dataset_name}_{train_noise_type}{'_noaug' if not with_aug else ''}"),
        train_dataloader=train_dataloader,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        train_cfg=dict(by_epoch=True, max_epochs=max_epoches, val_interval=val_interval),
        default_hooks=dict(checkpoint=dict(type=CheckpointHook, interval=save_cp_interval, save_best=["PixelRestoration/PSNR", "PixelRestoration/SSIM"], rule=["greater", "greater"])),
        resume=False,
        val_evaluator=[
            dict(type=PixelRestoration),
        ],
        val_dataloader=val_dataloader,
        val_cfg=dict(),
        load_from=ckpt_path,
        visualizer=visualizer,
        custom_hooks=None,
    )
    if is_clearml_installed:
        runner.visualizer.add_config(config=model_cfg)
        runner.visualizer.get_backend("ClearMLVisBackend")._task.add_tags([f"train:{train_dataset_name}/{train_noise_type}", f"test:{val_dataset_name}/{val_noise_type}", f"{model_arch_name}"])
    runner.train()


if __name__ == "__main__":
    start_one_training_process()
