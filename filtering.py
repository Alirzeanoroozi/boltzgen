from boltzgen.task.filter.filter import Filter

DESIGN_DIR = "outputs/test_design_rbx1_mine/inverse_folding/"
OUTDIR = "outputs/test_design_rbx1_mine/"

if __name__ == "__main__":
    task = Filter(
        design_dir=DESIGN_DIR,
        outdir=OUTDIR,
        budget=10,
        top_budget=10,
        use_affinity=False,
        filter_cysteine=True,
        from_inverse_folded=True,
        filter_bindingsite=False,
        modality="peptide",
        peptide_type="linear",
        alpha=0.01,
        random_state=0,
        metrics_override=None,
        num_liability_plots=0,
        plot_seq_logos=False,
        filter_designfolding=False,
        filter_biased=True,
        refolding_rmsd_threshold=2,
    )

    task.run()
    print("Done Filtering")
