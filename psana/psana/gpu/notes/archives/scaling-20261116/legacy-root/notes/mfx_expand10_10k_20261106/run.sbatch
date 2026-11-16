#!/usr/bin/env bash
#SBATCH --account=lcls
#SBATCH --partition=milano
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
stage_notes=${1:?absolute notes directory}
stage_runtime="$stage_notes/../mfx_expand10_stage2_20261105/runtime.sh"
bash "$stage_runtime" "$stage_notes/test_fixture.py"
exec bash "$stage_runtime" "$stage_notes/campaign.py" run
