This is a conversation with an LLM that suggested some tips for working on disk I/O constrained Slurm clusters, and specifically the Harvard FASRC cluster.

===============================================
Question: I often work on a Slurm cluster where everything moves very slow because of how bad the disk IO is. The network is amazing, and there are really big CPUs that I can access through interactive jobs, but disk IO is really slow.

What tools can I use to make my life easier on the Slurm cluster? In general, if I'm writing a Python script and things like that, how can I make my life better if I want to do things effectively and efficiently on this cluster?
===============================================

Great question — slow disk I/O on a shared filesystem (probably Lustre or GPFS?) is one of the most common pain points on HPC clusters. Here are the main strategies, roughly ordered by impact:

## Keep data in RAM or fast local storage

Most Slurm nodes have local scratch on SSDs or tmpfs mounted at something like `/tmp`, `/scratch/local`, or `$TMPDIR`. At the start of your job, copy what you need there, work from it, then copy results back. In your Slurm script:

```bash
cp $HOME/data/myfile.parquet $TMPDIR/
# ... do work reading/writing from $TMPDIR ...
cp $TMPDIR/results.csv $HOME/output/
```

For Python specifically, if your datasets fit in memory, load everything into memory once at the start and avoid repeated disk reads. Pandas, Polars, and PyArrow all support this well.

## Use better file formats

This matters a lot for the kind of tabular data you work with. Parquet and Feather/Arrow are columnar and compressed — they read dramatically faster than CSV on slow filesystems because there's simply less data to pull off disk. If you're still reading CSVs on the cluster, switching to Parquet alone could be a 5-10x speedup on load times. Polars in particular is excellent here since it supports lazy evaluation and predicate pushdown, meaning it only reads the columns and rows it actually needs.

## Reduce the number of file operations

Shared filesystems choke on metadata operations (opening/closing files, listing directories, stat calls). A few concrete things:

- **Bundle small files** into archives, databases, or single large files. If you have thousands of small CSVs, consolidate them into one Parquet or an SQLite database before running your job.
- **Avoid `os.listdir()` or `glob.glob()` on huge directories** — cache the file list once.
- **Use `pip install --cache-dir $TMPDIR`** or install into a conda environment that lives on local scratch. Python imports hit the filesystem hard (hundreds of small `.pyc` files), which is brutal on slow I/O.

## Package your Python environment smartly

This is an underrated one. Every time Python imports a package, it stats and reads dozens of files. Options:

- **conda-pack** or **venv-pack**: bundle your entire environment into a tarball, then unpack it to local scratch at job start. This is probably the single biggest quality-of-life improvement if you're noticing slow startup times.
- **Containers**: Singularity/Apptainer images keep your whole environment in one file, which is much friendlier to shared filesystems. Since your cluster has good network, pulling images is fast.

```bash
# In your Slurm script
tar -xzf ~/envs/myenv.tar.gz -C $TMPDIR/env
source $TMPDIR/env/bin/activate
```

## Use memory-mapped I/O or streaming where appropriate

For large arrays, `numpy.memmap` or HDF5 via `h5py` let the OS page in only what's needed. For very large tabular data, DuckDB can query Parquet files with minimal memory overhead and handles I/O quite efficiently — it's worth considering as a replacement for loading entire DataFrames when you only need aggregations or subsets.

## Leverage the good network

Since your network is fast, consider keeping data on a remote store and streaming it rather than relying on the parallel filesystem. For example, if your cluster or lab has an S3-compatible object store (MinIO, etc.), tools like `fsspec`, `s3fs`, or `pyarrow.fs` can read Parquet directly from it. The network might actually beat the shared disk.

## Practical workflow tips

- **Interactive development**: if you're iterating on code, do your editing and small tests locally or on a login node with tiny data samples, and only submit to the cluster for full runs. This avoids the "wait 30 seconds for vim to save" problem.
- **`squeue`, `sacct`**: pipe these to local files or use `watch` rather than hammering the scheduler.
- **Snakemake** (which you already use): it does a lot of filesystem stat calls to check rule freshness. On slow I/O this can be painful — consider using `--rerun-triggers mtime` and keeping your Snakemake working directory on faster storage if possible.

The highest-ROI moves for your situation are probably: (1) conda-pack your environment to local scratch, (2) switch any remaining CSVs to Parquet, and (3) stage input data to `$TMPDIR` at job start. Those three alone usually make a night-and-day difference.

===============================================
Follow up Question: It’s the Harvard FASRC cluster. Look up some info specific to that?
===============================================

Here's what's specific to Cannon (the FASRC cluster):

## The storage hierarchy you should know

Cannon has several distinct filesystems, and which one you use for what matters enormously:

**`/n/netscratch` (network scratch, `$SCRATCH`)** — This is the VAST filesystem that FASRC recommends as your primary job working area, as it's highly optimized for cluster use. Files are purged after 90 days and there's no backup, but this is where your active computation should happen. Create a folder under your lab group's space there.

**`/scratch` (local on-node scratch)** — This is the real secret weapon. It's local on-node storage, meaning it's on the node's local hard drive — very fast and won't interfere with I/O on other nodes. For batch jobs, staging data here at the start of your job and working from it gives you the best possible I/O performance. This is what I was calling `$TMPDIR` in my earlier advice.

**Home directory (`~`)** and **lab storage (`/n/holylabs/...`)** — Both have moderate performance and are not appropriate for I/O intensive or large numbers of jobs. Use these for code, environments, and results — not for active computation.

## A known gotcha: VAST and HDF5

This is important for your research data work — VAST filesystems (netscratch and holylabs) have known issues with file locking for HDF5 and other things that do sophisticated parallel I/O. If you're using HDF5, either work from `/scratch` (local node storage) or be aware you may hit locking errors on the VAST filesystems. Parquet doesn't have this issue, which is another reason to prefer it.

## Conda/Mamba environments on Cannon

FASRC uses Mamba as their package manager (it's a faster drop-in replacement for conda). A few Cannon-specific practices:

**Put environments on lab storage, not home.** For optimal performance, use the `--prefix` option to create your conda environments to your lab space:

```bash
module load python
mamba create --prefix=/n/holylabs/LABS/<your_PI_lab>/Lab/conda/myenv python=3.11 pandas polars pyarrow
source activate /n/holylabs/LABS/<your_PI_lab>/Lab/conda/myenv
```

This keeps your 100GB home directory quota clear and lets labmates share environments.

**Always create environments from an interactive session**, not the login node:

```bash
salloc --partition test --nodes=1 --cpus-per-task=2 --mem=4GB --time=0-02:00:00
module load python
mamba create ...
```

**Avoid pip outside of a mamba environment.** Running pip install outside of a mamba environment places packages in `$HOME/.local`, which can lead to package conflicts and cause packages to fail to install or load correctly via mamba.

**Don't put `conda initialize` in your `.bashrc`.** The conda initialize section is known to create issues on the FASRC clusters.

## The directory file count rule

This is a hard rule on Cannon: keep the number of files in a single directory under 1,000 — larger numbers can cause latency issues, and numbers much larger may cause system-wide issues. This is especially relevant when you're generating outputs from batch jobs. Structure your output directories with subdirectories rather than dumping thousands of files flat.

## Practical template for your Slurm jobs

Combining all of this, a well-structured batch script for your typical data processing work might look like:

```bash
#!/bin/bash
#SBATCH -p shared          # or serial_requeue for better throughput
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-04:00

# Load environment
module load python
source activate /n/holylabs/LABS/hausmann_lab/Lab/conda/myenv

# Stage data to fast local scratch
mkdir -p /scratch/mywork_${SLURM_JOB_ID}
cp /n/netscratch/hausmann_lab/mydata/*.parquet /scratch/mywork_${SLURM_JOB_ID}/

# Run from local scratch
cd /scratch/mywork_${SLURM_JOB_ID}
python /n/home.../myscript.py --input . --output ./results/

# Copy results back
cp -r ./results/ /n/netscratch/hausmann_lab/myresults/

# Clean up
rm -rf /scratch/mywork_${SLURM_JOB_ID}
```

The key pattern is: read from `/scratch` (local), write to `/scratch` (local), copy back to netscratch at the end. This avoids hammering the shared filesystem during your actual computation.

## For `serial_requeue` — checkpoint your work

If you use `serial_requeue` (which you should for better throughput — it can scavenge idle resources from lab-owned partitions), your job may be stopped and restarted elsewhere, so checkpoint your code to let a requeued job pick up where it left off. For Python, this means periodically writing intermediate results and being able to resume from them.
