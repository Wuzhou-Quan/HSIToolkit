import gsettings, torch, os, mmengine, torchvision.transforms as v2, click
from inad_hsi_toolkit.metrics.mmengine_metrics import PixelRestoration
from mmengine.config import Config
from rich.progress import track
from inad_hsi_toolkit.miscs.mme_general_trainer_wrapper import TrainingWrapper


# Example (HSDT_S gaussian):
# python val.py --model_arch_name hsdt_s --dataset_name icvl_test --noise_types gaussian30_70blind,gaussian30,gaussian50,gaussian70,gaussian10_70blind --ckpt_path checkpoints/hsdt_s_gaussian.pth --num_bands 31


@click.command()
@click.option(
    "--model_arch_name",
    prompt="Enter the model architecture name from the deployments folder",
    type=str,
    required=True,
    help="Specify the architecture name of the model located in the deployments folder.",
)
@click.option(
    "--dataset_name",
    prompt="Enter the name of the validation dataset from the deployments folder",
    type=str,
    required=True,
    help="Specify the name of the validation dataset located in the deployments folder.",
)
@click.option(
    "--noise_types",
    prompt="Enter the noise types (separated by commas, e.g., 'gaussian30,gaussian50')",
    type=str,
    required=True,
    help="Provide a list of noise types you want to test, separated by commas.",
)
@click.option("--ckpt_path", prompt="Enter the checkpoint file path", type=str, required=True, help="Specify the file path to the model checkpoint.")
@click.option("--num_bands", prompt="Enter the number of bands (default is 31)", type=int, required=True, default=31, help="Specify the number of bands (default value is 31 if not provided).")
def start_one_validation_process(model_arch_name: str, dataset_name: str, noise_types: str, ckpt_path: str, num_bands: int):
    # Model
    model_deployment_file_path = gsettings.deployment_path / "models" / f"{model_arch_name}.py"
    model_cfg = Config.fromfile(
        os.fspath(model_deployment_file_path),
        lazy_import=False,
    )

    def replace_numbands(cfg, numbands):
        for k, v in cfg.items():
            if isinstance(v, dict):
                replace_numbands(v, numbands)
            elif v == "%numbands%":
                cfg[k] = numbands

    replace_numbands(model_cfg, num_bands)
    model: torch.nn.Module = mmengine.MODELS.build(model_cfg.model).cuda()
    ckpt = torch.load(os.fspath(ckpt_path))
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    if isinstance(model, TrainingWrapper):
        for k in list(ckpt.keys()):
            if not k.startswith("model."):
                ckpt[f"model.{k}"] = ckpt.pop(k)
    model.load_state_dict(ckpt)
    model.eval()

    # Dataset
    dataset_deployment_file_path = gsettings.deployment_path / "datasets" / f"{dataset_name}.py"
    dataset_cfg = Config.fromfile(
        os.fspath(dataset_deployment_file_path),
        lazy_import=False,
    )
    dataset_cfg.dataset["augments"] = v2.Compose(
        [
            v2.ToTensor(),
            v2.ConvertImageDtype(torch.float32),
        ]
    )
    noise_types = noise_types.split(",")
    for noise_type in noise_types:
        # Noise
        if noise_type.upper() != "NONE":
            noise_maker_file_path = gsettings.deployment_path / "noise_makers" / f"{noise_type}.py"
            dataset_cfg.dataset["noise_maker"] = Config.fromfile(
                os.fspath(noise_maker_file_path),
                lazy_import=False,
            ).noise_maker
        else:
            dataset_cfg.dataset["noise_maker"] = None
        val_dataset = mmengine.DATASETS.build(dataset_cfg.dataset)
        metrics = PixelRestoration("cuda")
        src_metrics = PixelRestoration("cuda")
        with torch.no_grad():
            for i in track(range(len(val_dataset)), description=f"{dataset_name} with {noise_type} noise"):
                val_sample = val_dataset[i]
                x = val_sample["x"].unsqueeze(0).cuda()
                gt = val_sample["gt"].unsqueeze(0).cuda()
                y = model(x)
                src_metrics.process({"gt": gt}, x)
                metrics.process({"gt": gt}, y)
        print(metrics.compute_metrics(src_metrics.results))
        print(metrics.compute_metrics(metrics.results))


if __name__ == "__main__":
    start_one_validation_process()
