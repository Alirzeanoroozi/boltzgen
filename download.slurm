# Choose any location; this is default in yaml files
mkdir -p training_data
cd training_data

# ─ Targets ─
# wget -O targets.zip "https://huggingface.co/datasets/boltzgen/boltzgen1_train/resolve/main/targets.zip?download=true"
unzip targets.zip      # → training_data/targets/

# ─ MSAs  ─
# wget -O msa.zip "https://huggingface.co/datasets/boltzgen/boltzgen1_train/resolve/main/msa.zip?download=true"
unzip msa.zip          # → training_data/msa/

# ─ Small-molecule dictionary ─
# wget -O mols.zip "https://huggingface.co/datasets/boltzgen/inference-data/resolve/main/mols.zip?download=true"
unzip mols.zip         # → training_data/mols/

# ─ Folding checkpoint  ─
# wget -O boltz2_fold.ckpt "https://huggingface.co/boltzgen/boltzgen-1/resolve/main/boltz2_conf_final.ckpt?download=true"

# ─────────── (optional) pretrained structure-only ckpt ───────────
# Needed ONLY if you want to resume from a structure-trained model.
# wget -O boltzgen1_structuretrained_small.ckpt \
#  "https://huggingface.co/boltzgen/boltzgen-1/resolve/main/boltzgen1_structuretrained_small.ckpt?download=true"