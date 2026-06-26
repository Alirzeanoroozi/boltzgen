from boltzgen.task.analyze.analyze import Analyze
from boltzgen.task.predict.data_from_generated import FromGeneratedDataModule, DataConfig
from boltzgen.data.tokenize.tokenizer import Tokenizer
from boltzgen.data.feature.featurizer import Featurizer

MOL_DIR = "/home/anoroozi25/.cache/huggingface/hub/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip"
DESIGN_DIR = "outputs/test_design_rbx1_mine/inverse_folding/"

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
        num_workers=4,
        pin_memory=True,
        disulfide_prob=1.0,
        disulfide_on=True,
    )

    data = FromGeneratedDataModule(
        data_config,
        design_dir=DESIGN_DIR,
        target_templates=False,
        return_native=False,
        fail_if_no_designs=True,
        skip_existing=False,
        skip_existing_kind="analyzed",
    )

    task = Analyze(
        name="analyze",
        data=data,
        design_dir=DESIGN_DIR,
        debug=False,
        num_processes=32,
        affinity_metrics=False,
        backbone_fold_metrics=True,
        noncovalents_original=False,
        noncovalents_refolded=True,
        delta_sasa_original=False,
        delta_sasa_refolded=True,
        largest_hydrophobic=False,
        largest_hydrophobic_refolded=False,
        run_clustering=False,
        liability_analysis=True,
        liability_modality="peptide",
        liability_peptide_type="linear",
        diversity_original=False,
        diversity_refolded=False,
        diversity_per_target_original=False,
        diversity_per_target_refolded=False,
        novelty_original=False,
        novelty_refolded=False,
        novelty_per_target_original=False,
        novelty_per_target_refolded=False,
        ss_conditioning_metrics=False,
        sequence_recovery=False,
        native=False,
        compute_lddts=False,
        designfolding_metrics=False,
        allatom_fold_metrics=False,
    )

    task.run()
    print("Done Analysis")
