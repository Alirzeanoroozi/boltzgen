import torch
torch.set_grad_enabled(False)

from pytorch_lightning import LightningModule, Trainer

from src.task.predict.data_from_yaml import FromYamlDataModule, DataConfig
from src.data.tokenize.tokenizer import Tokenizer
from src.data.feature.featurizer import Featurizer
from src.task.predict.writer import DesignWriter
from src.model.models.boltz import Boltz

MOL_DIR = "/home/anoroozi25/.cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
CHECKPOINT = "/home/anoroozi25/.cache/huggingface/hub/models--boltzgen--boltzgen-1/snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltzgen1_diverse.ckpt"

OUTPUT_DIR = "hallucination_run/"
YAML_PATH = "hallucination.yaml"


if __name__ == "__main__":
    print("Starting Generate...")
        
    data_config = DataConfig(
        moldir=MOL_DIR,
        multiplicity=10,
        yaml_path=YAML_PATH,
        tokenizer=Tokenizer(
            atomize_modified_residues=False,
            map_to_closest_residue=False
        ),
        featurizer=Featurizer(), # TODO: Add featurizer from config
        backbone_only=False,
        atom14=True,
        atom37=False,
        design=True,
        compute_affinity=False,
        disulfide_prob=1.0,
        disulfide_on=True,
        skip_existing=False,
        skip_offset=0,
        diffusion_samples=10,
        output_dir=OUTPUT_DIR
    )
    
    data = FromYamlDataModule(
        data_config,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        extra_features=None # TODO: Add extra features from config
    )
    
    writer = DesignWriter(
        output_dir=OUTPUT_DIR,
        res_atoms_only=False,
        atom14=True,
        atom37=False,
        backbone_only=False,
        write_native=False
    )
    
    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 500,
        "diffusion_samples": 10,
    }

    # Load model
    model_module: LightningModule = Boltz.load_from_checkpoint(
        CHECKPOINT,
        strict=True,
        use_ema=False,
        checkpoint_diffusion_conditioning=False,
        map_location="cpu",
        predict_args=predict_args,
        weights_only=False,
    )
    model_module.eval()

    lightning_trainer = Trainer(
        default_root_dir=OUTPUT_DIR,
        strategy="auto",
        callbacks=writer,
        logger=False,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
    )

    # Run training
    lightning_trainer.predict(
        model_module,
        datamodule=data,
        return_predictions=True
    )
    
    print("Done")