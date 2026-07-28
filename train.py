import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy

from boltzgen.data.crop.multimer import MultimerCropper
from boltzgen.data.feature.featurizer import Featurizer
from boltzgen.data.filter.dynamic.date import DateFilter
from boltzgen.data.filter.dynamic.resolution import ResolutionFilter
from boltzgen.data.filter.dynamic.size import SizeFilter
from boltzgen.data.sample.cluster import ClusterSampler
from boltzgen.data.select.protein import ProteinSelector
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.model.models.boltz import Boltz
from boltzgen.model.validation.design import DesignValidator
from boltzgen.task.predict.data_from_generated import (
    DataConfig as GeneratedDataConfig,
    FromGeneratedDataModule,
)
from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.train.data import DatasetConfig, DataConfig, TrainingDataModule
from boltzgen.model.validation.refolding import RefoldingValidator

NAME = "boltzgen_small"
OUTPUT = "workdir"
FOLDING_CHECKPOINT = "./training_data/boltz2_fold.ckpt"
MOL_DIR = "./training_data/mols"
TARGET_DIR = "./training_data/targets"
MSA_DIR = "./training_data/msa"

SPLIT = "./boltzgen/resources/splits_small/validation_ids_boltz2_all.txt"
MONOMER_SPLIT = "./boltzgen/resources/splits_small/val_monomers_boltzgen_min50_max220.txt"
LIGAND_SPLIT = "./boltzgen/resources/splits_small/val_ccd_pdb_pairs_boltzgen.txt"

CONFIDENCE_PREDICTION = False
BACKBONE_ONLY = False
ATOM14 = True
ATOM37 = False

def build_data() -> DataConfig:
    selector = ProteinSelector(
        design_neighborhood_sizes=[2, 4, 6, 8, 10, 12, 14, 16, 18],
        substructure_neighborhood_sizes=[2, 4, 6, 8, 10, 12, 24],
        structure_condition_prob=0.4,
        distance_noise_std=1,
        run_selection=True,
        specify_binding_sites=True,
        ss_condition_prob=0.1,
        select_all=False,
        chain_reindexing=False,
    )

    dataset = DatasetConfig(
        target_dir=TARGET_DIR,
        msa_dir=MSA_DIR,
        prob=1,
        filters=[
            SizeFilter(min_chains=1, max_chains=2),
            DateFilter(date="2022-06-01", ref="released"),
            ResolutionFilter(resolution=9.0),
        ],
        sampler=ClusterSampler(),
        cropper=MultimerCropper(
            neighborhood_sizes=list(range(2, 41, 2)),
        ),
        split=SPLIT,
        symmetry_correction=False,
        val_group="RCSB",
    )

    return DataConfig(
        datasets=[dataset],
        tokenizer=Tokenizer(atomize_modified_residues=False),
        featurizer=Featurizer(),
        selector=selector,
        moldir=MOL_DIR,
        max_tokens=256,
        max_atoms=2048,
        max_seqs=1024,
        pad_to_max_tokens=True,
        pad_to_max_atoms=True,
        pad_to_max_seqs=True,
        samples_per_epoch=100000,
        batch_size=1,
        num_workers=4,
        random_seed=42,
        pin_memory=True,
        overfit=None,
        return_train_symmetries=False,
        return_val_symmetries=False,
        atoms_per_window_queries=32,
        min_dist=2.0,
        max_dist=22.0,
        num_bins=64,
        single_sequence_prop_training=0.1,
        msa_sampling_training=True,
        design=True,
        backbone_only=BACKBONE_ONLY,
        atom14=ATOM14,
        atom37=ATOM37,
        monomer_split=MONOMER_SPLIT,
        monomer_target_dir=TARGET_DIR,
        monomer_target_structure_condition=True,
        monomer_seq_len=100,
        ligand_split=LIGAND_SPLIT,
        ligand_target_dir=TARGET_DIR,
        ligand_seq_len=100,
    )

def build_refolding_validator():
    analyze_data = FromGeneratedDataModule(
        cfg=GeneratedDataConfig(
            tokenizer=Tokenizer(atomize_modified_residues=False),
            featurizer=Featurizer(),
            suffix=".cif",
            suffix_metadata=".npz",
            suffix_native="_native.cif",
            samples_per_target=1,
            num_targets=100000000,
            moldir=MOL_DIR,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
        ),
        target_templates=True,
        return_native=True,
    )

    analyze_task = Analyze(
        name=NAME,
        design_dir=None,
        num_processes=1,
        affinity_metrics=False,
        allatom_fold_metrics=True,
        backbone_fold_metrics=True,
        noncovalents_original=False,
        noncovalents_refolded=False,
        delta_sasa_original=False,
        delta_sasa_refolded=False,
        largest_hydrophobic=False,
        largest_hydrophobic_refolded=False,
        run_clustering=False,
        liability_analysis=False,
        liability_modality="peptide",
        liability_peptide_type="linear",
        diversity_original=True,
        diversity_refolded=True,
        diversity_per_target_original=False,
        diversity_per_target_refolded=False,
        novelty_original=False,
        novelty_refolded=False,
        novelty_per_target_original=False,
        novelty_per_target_refolded=False,
        data=analyze_data,
    )

    return RefoldingValidator(
        val_names=["RCSB"],
        step_scale=1.5,
        noise_scale=0.75,
        atom14=ATOM14,
        atom37=ATOM37,
        backbone_only=BACKBONE_ONLY,
        val_monomer=MONOMER_SPLIT,
        val_ligand=LIGAND_SPLIT,
        analyze_task=analyze_task,
        folding_checkpoint=FOLDING_CHECKPOINT,
        folding_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
        },
        folding_model_args={"validators": None},
    )


class DotDict(dict):
    """DotDict is a dictionary with attribute-style access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc



def build_model() -> Boltz:
    design_validator = DesignValidator(
        val_names=["RCSB"],
        confidence_prediction=CONFIDENCE_PREDICTION,
        backbone_only=BACKBONE_ONLY,
        atom14=ATOM14,
        atom37=ATOM37,
    )

    return Boltz(
        atom_s=128,
        atom_z=16,
        token_s=384,
        token_z=128,
        num_bins=64,
        training_args=DotDict({
            "recycling_steps": 3,
            "sampling_steps": 20,
            "diffusion_multiplicity": 12,
            "diffusion_samples": 1,
            "confidence_loss_weight": 1e-4,
            "diffusion_loss_weight": 4.0,
            "distogram_loss_weight": 3e-2,
            "bfactor_loss_weight": 1e-3,
            "res_type_loss_weight": 3e-2,
            "adam_beta_1": 0.9,
            "adam_beta_2": 0.95,
            "adam_eps": 0.00000001,
            "lr_scheduler": "af3",
            "base_lr": 0.0,
            "max_lr": 0.0018,
            "lr_warmup_no_steps": 1000,
            "lr_start_decay_after_n_steps": 50000,
            "lr_decay_every_n_steps": 50000,
            "lr_decay_factor": 0.95,
            "weight_decay": 0.003,
            "weight_decay_exclude": True,
            "skip_batch_by_single_rep": True
        }),
        validation_args=DotDict({
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "symmetry_correction": False,
        }),
        embedder_args={
            "atom_encoder_depth": 3,
            "atom_encoder_heads": 4,
            "add_mol_type_feat": True,
            "add_method_conditioning": True,
            "add_modified_flag": True,
            "add_cyclic_flag": True,
            "add_design_mask_flag": True,
            "add_binding_specification": True,
            "add_ss_specification": True,
        },
        msa_args={
            "msa_s": 64,
            "msa_blocks": 3,
            "msa_dropout": 0.15,
            "z_dropout": 0.25,
            "miniformer_blocks": True,
            "pairwise_head_width": 32,
            "pairwise_num_heads": 4,
            "use_paired_feature": True,
            "activation_checkpointing": False,
        },
        pairformer_args={
            "num_blocks": 12,
            "num_heads": 16,
            "dropout": 0.25,
            "post_layer_norm": False,
            "activation_checkpointing": False,
        },
        score_model_args={
            "sigma_data": 16,
            "dim_fourier": 256,
            "atom_encoder_depth": 3,
            "atom_encoder_heads": 4,
            "token_layers": 1,
            "token_transformer_depth": 8,
            "token_transformer_heads": 16,
            "diffusion_pairformer_args": {
                "num_blocks": 0,
                "num_heads": 2,
                "dropout": 0,
                "use_s_to_z": False,
            },
            "atom_decoder_depth": 3,
            "atom_decoder_heads": 4,
            "conditioning_transition_layers": 2,
            "transformer_post_ln": False,
            "activation_checkpointing": False,
        },
        diffusion_process_args={
            "sigma_min": 0.0004,
            "sigma_max": 160.0,
            "sigma_data": 16.0,
            "rho": 7,
            "P_mean": -1.2,
            "P_std": 1.5,
            "gamma_0": 0.8,
            "gamma_min": 1.0,
            "noise_scale": 1.0,
            "step_scale": 1.0,
            "mse_rotational_alignment": True,
            "coordinate_augmentation": True,
            "alignment_reverse_diff": True,
            "synchronize_sigmas": False,
        },
        diffusion_loss_args={
            "add_smooth_lddt_loss": True,
            "add_bond_loss": False,
            "nucleotide_loss_weight": 5.0,
            "ligand_loss_weight": 10.0,
        },
        validators=[design_validator],
        masker_args={
            "mask": True,
            "mask_backbone": False,
            "mask_disto": True,
        },
        atom_feature_dim=388,
        template_args={
            "template_dim": 64,
            "template_blocks": 2,
            "miniformer_blocks": True,
            "activation_checkpointing": False,
        },
        use_miniformer=True,
        confidence_prediction=CONFIDENCE_PREDICTION,
        structure_prediction_training=True, # most Important!
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        exclude_ions_from_lddt=True,
        ema=True,
        ema_decay=0.999,
        num_val_datasets=1,
        aggregate_distogram=True,
        bond_type_feature=True,
        predict_bfactor=True,
        predict_res_type=False,
        checkpoint_diffusion_conditioning=False,
        use_kernels=True,
        freeze_template_weights=True,
        use_templates=True,
        use_token_distances=True,
        token_distance_args={
            "token_distance_dim": 64,
            "token_distance_blocks": 2,
            "use_token_distance_feats": True,
            "distance_gaussian_dim": 32,
        },
        refolding_validator=build_refolding_validator(),
    )

if __name__ == "__main__":
    data_config = build_data()
    data_module = TrainingDataModule(data_config)
    model_module = build_model()

    dirpath = f"{OUTPUT}/{NAME}"
    os.makedirs(dirpath, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            monitor="val/lddt",
            save_top_k=-1,
            save_last=True,
            mode="max",
            every_n_epochs=1,
        )
    ]

    loggers = [
        WandbLogger(
            name=NAME,
            save_dir=dirpath,
            project="boltzgen",
        )
    ]

    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=DDPStrategy(
            cluster_environment=LightningEnvironment(),
            find_unused_parameters=True,
        ),
        callbacks=callbacks,
        logger=loggers,
        accelerator="gpu",
        devices=7,
        precision="bf16-mixed",
    )

    trainer.fit(model_module, datamodule=data_module)

    trainer.validate(model_module, datamodule=data_module)
