###############################################
# 🔧 Configuration File: config.sh
# Update the paths below **after** downloading the required files.
# These paths ensure correct loading of data, patches, and pretrained models.
###############################################

# 📂 Main data directory
# Absolute path where all generated data and pretrained models are stored.
# 🔺 Make sure to place the pretrained models inside this directory.
Main_Data_Path=

# 🖼️ Validation image directory
# Path to the downloaded validation images used for evaluating the distilled data.
# You may store them anywhere, just keep this path updated accordingly.
val_dir=

# 🧩 Initialized patch directory
# Path to the downloaded initialized patch files (from Google Drive).
# Flexible placement, just ensure the path is correctly specified here.
patch_dir=

###############################################
# ⚠️ Reminder:
# - Pretrained models MUST be placed under `${Main_Data_Path}`.
# - No strict location is required for `val_dir` or `patch_dir`, 
###############################################
