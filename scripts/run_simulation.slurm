#!/bin/bash
#SBATCH -J StabilityAnalysis
#SBATCH --mail-user npradeep96
#SBATCH -p sched_mit_arupc_long
#SBATCH -t 12:00:00
#SBATCH --mem-per-cpu 8000
cd "$SLURM_SUBMIT_DIR"
echo $PWD

stage_parameters()
{
    cp $input_file input_parameters_$SLURM_JOBID.txt
}

run_program()
{
    source activate fipyenv
    python run_simulation.py --i $input_file --o $out_folder
    conda deactivate
}

cleanup_files()
{
    rm input_parameters_$SLURM_JOBID.txt
    echo "DONE"
}

stage_parameters
run_program
cleanup_files
