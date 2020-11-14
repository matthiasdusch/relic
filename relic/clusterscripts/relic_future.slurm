#!/bin/bash
#
#SBATCH --export=NONE
#SBATCH --job-name=commitment
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --mail-user=matthias.dusch@uibk.ac.at
#SBATCH --mail-type=ALL
#SBATCH --array=1-30
#SBATCH --qos=low

# Abort whenever a single step fails. Without this, bash will just continue on errors.
set -e

echo "$SLURM_ARRAY_TASK_ID"
JOBID=$SLURM_ARRAY_TASK_ID
export JOBID

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
OGGM_WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/$JOBID"
mkdir -p "$OGGM_WORKDIR"
echo "Workdir for this run: $OGGM_WORKDIR"

# Export the WORKDIR as environment variable so our script can use it to find its working directory.
export OGGM_WORKDIR

# Use the local data download cache
export OGGM_DOWNLOAD_CACHE=/home/data/download
export OGGM_DOWNLOAD_CACHE_RO=1
export OGGM_EXTRACT_DIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/oggm_tmp"

# All commands in the EOF block run inside of the container
# Adjust container version to your needs, they are guaranteed to never change after their respective day has passed.
singularity exec /home/users/mdusch/oggm_20201023.sif bash -s <<EOF
  set -e

  # Setup a fake home dir inside of our workdir, so we don't clutter the actual shared homedir with potentially incompatible stuff.
  export HOME="$OGGM_WORKDIR/fake_home"
  mkdir "\$HOME"

  # Create a venv that _does_ use system-site-packages, since everything is already installed on the container.
  # We cannot work on the container itself, as the base system is immutable.
  python3 -m venv --system-site-packages "$OGGM_WORKDIR/oggm_env_singu"
  source "$OGGM_WORKDIR/oggm_env_singu/bin/activate"

  # Make sure latest pip is installed
  pip install --upgrade pip setuptools

  # Install a fixed OGGM version ()
  # pip install --upgrade "git+https://github.com/OGGM/oggm.git@ab702a307906bd43a6d11c73ffdd939bbfbbbf8e"

  # install a fixed relic version
  cd "$OGGM_WORKDIR/fake_home"
  git clone https://github.com/matthiasdusch/relic.git
  cd relic
  git reset --hard 69760720d23ddae3eaf8444b2f2fb24ee6494883
  pip install -e .


  # Pre download the files to be sure
  python3 -c "from salem.utils import get_demo_file; print('Salem test file: ' + get_demo_file('world_borders.shp'))"
  python3 -c "from oggm.utils import get_demo_file; print('OGGM test file: ' + get_demo_file('srtm_oetztal.tif'))"

  # Finally, the runs
  python3 "$HOME/length_change/finito/relic_future.py"
EOF

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"

