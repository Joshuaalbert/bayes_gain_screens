#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $progname --obs_num [--archive_dir --root_working_dir --script_dir --region_file]

   optional arguments:
     -h, --help           show this help message and exit

HEREDOC
}

# initialize variables
progname=$(basename $0)
obs_num=
archive_dir=${HOME}/store/P126+65
root_working_dir=${HOME}/store/root_chap3
script_dir=${HOME}/store/scripts
region_file=None
mount_dirs=/beegfs/lofar
#${HOME}/store/lockman/LHdeepbright.reg

# use getopt and store the output into $OPTS
# note the use of -o for the short options, --long for the long name options
# and a : for any option that takes a parameter
OPTS=$(getopt -o "h" --long "help,obs_num:,archive_dir:,root_working_dir:,script_dir:,region_file:,mount_dirs:" -n "$progname" -- "$@")
if [ $? != 0 ] ; then echo "Error in command line arguments." >&2 ; usage; exit 1 ; fi
eval set -- "$OPTS"

while true; do
  # uncomment the next line to see how shift is working
  # echo "\$1:\"$1\" \$2:\"$2\""
  case "$1" in
    -h | --help ) usage; exit; ;;
    --obs_num ) obs_num="$2"; shift 2 ;;
    --archive_dir ) archive_dir="$2"; shift 2 ;;
    --root_working_dir ) root_working_dir="$2"; shift 2 ;;
    --script_dir ) script_dir="$2"; shift 2 ;;
    --region_file ) region_file="$2"; shift 2 ;;
    --mount_dirs ) mount_dirs="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -z "$obs_num" ]
then
    usage;
    exit;
fi

#if [ -z "$PBS_O_WORKDIR" ] || [ -z "$PBS_JOBNAME" ] || [ -z "$PBS_JOBID" ]
#then
#    if [ -z "$SLURM_SUBMIT_DIR" ] || [ -z "$SLURM_JOB_NAME" ] || [ -z "$SLURM_JOB_ID" ]
#    then
#        # using unknown queue system
#        log=./job_unnamed.log
#    else
#        # using slurm system
#        log="$SLURM_SUBMIT_DIR"/job_"$SLURM_JOB_NAME"_"$SLURM_JOB_ID".log
#    fi
#else
#    # using pbs system
#    log="$PBS_O_WORKDIR"/job_"$PBS_JOBNAME"_"$PBS_JOBID".log
#fi



source ~/.bashrc
#ddf_singularity

singularity exec -B /tmp,/dev/shm ${HOME}/store/lofar_sksp_ddf.simg CleanSHM.py
#&> "$log"

singularity exec -B /tmp,/dev/shm,${HOME},${mount_dirs} ${HOME}/store/lofar_sksp_ddf.simg \
    python ${HOME}/store/scripts/pipeline.py \
        --archive_dir="$archive_dir" \
        --root_working_dir="$root_working_dir" \
        --script_dir="$script_dir" \
        --region_file="$region_file" \
        --ref_dir=0 \
        --ncpu=32 \
        --block_size=50 \
        --deployment_type=directional \
        --no_subtract=False \
        --do_choose_calibrators=2 \
        --do_subtract=2 \
        --do_solve_dds4=2 \
        --do_smooth_dds4=2 \
        --do_slow_dds4=2 \
        --do_tec_inference=2 \
        --do_infer_screen=2 \
        --do_merge_slow=2 \
        --obs_num="$obs_num"
#         &>> "$log"

singularity exec -B /tmp,/dev/shm ${HOME}/store/lofar_sksp_ddf.simg CleanSHM.py
# &>> "$log"

singularity exec -B /tmp,/dev/shm,${HOME},${mount_dirs} ${HOME}/store/lofar_sksp_ddf_gainscreens_premerge.simg \
    python ${HOME}/store/scripts/pipeline.py \
        --archive_dir="$archive_dir" \
        --root_working_dir="$root_working_dir" \
        --script_dir="$script_dir" \
        --region_file="$region_file" \
        --ref_dir=0 \
        --ncpu=32 \
        --block_size=10 \
        --deployment_type=directional \
        --do_image_smooth=2 \
        --do_image_dds4=2 \
        --do_image_smooth_slow=2 \
        --do_image_screen_slow=0 \
        --do_image_screen=0 \
        --obs_num="$obs_num"
#        &>> "$log"
