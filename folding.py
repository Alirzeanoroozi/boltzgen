import torch
torch.set_grad_enabled(False)

from pytorch_lightning import Trainer

from boltzgen.task.predict.data_from_generated import FromGeneratedDataModule, DataConfig
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.task.predict.writer import FoldingWriter
from boltzgen.model.models.boltz import Boltz

MOL_DIR = "/home/anoroozi25/.cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
CHECKPOINT = "/home/anoroozi25/.cache/huggingface/hub/models--boltzgen--boltzgen-1/snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltz2_conf_final.ckpt"

DESIGN_DIR = "outputs/test_design_rbx1_mine/inverse_folding/"

KEYS_DICT_OUT = [
    "min_interaction_pae",
    "min_design_to_target_pae",
    "interaction_pae",
    "ligand_iptm",
    "protein_iptm",
    "iptm",
    "design_iptm",
    "design_iiptm",
    "design_to_target_iptm",
    "design_ptm",
    "target_ptm",
    "ptm",
]

if __name__ == "__main__":
    data_config = DataConfig(
        moldir=MOL_DIR,
        tokenizer=Tokenizer(atomize_modified_residues=False),
        featurizer=Featurizer(),
        suffix=".cif",
        suffix_metadata=".npz",
        suffix_native="_native.cif",
        samples_per_target=10**15,
        num_targets=10**13,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        disulfide_prob=1.0,
        disulfide_on=True,
    )

    data = FromGeneratedDataModule(
        data_config,
        design_dir=DESIGN_DIR,
        target_templates=True,
        return_native=False,
        fail_if_no_designs=True,
        output_dir=None,
        skip_existing=False,
        skip_existing_kind="folded",
    )

    writer = FoldingWriter(design_dir=DESIGN_DIR)

    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 5,
        "keys_dict_out": KEYS_DICT_OUT,
    }

    override = {
        "validators": None,
        "use_kernels": True,
    }

    model_module = Boltz.load_from_checkpoint(
        CHECKPOINT,
        strict=True,
        use_ema=False,
        checkpoint_diffusion_conditioning=False,
        map_location="cpu",
        predict_args=predict_args,
        weights_only=False,
        **override,
    )
    model_module.eval()

    lightning_trainer = Trainer(
        strategy="auto",
        callbacks=writer,
        logger=False,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
    )

    lightning_trainer.predict(
        model_module,
        datamodule=data,
        return_predictions=False,
    )

    print("Done Folding")
