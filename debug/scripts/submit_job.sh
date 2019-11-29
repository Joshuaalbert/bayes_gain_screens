#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $progname --obs_num [--archive_dir --root_working_dir --script_dir --region_file --mount_dirs --ncpu]

   optional arguments:
     -h, --help           show this help message and exit

HEREDOC
}

# initialize variables and defaults
progname=$(basename $0)
obs_num=562061
archive_dir=${HOME}/store/P126+65
root_working_dir=${HOME}/store/root
script_dir=${HOME}/store/scripts
region_file=None
mount_dirs=/beegfs/lofar
ncpu=24

###
# calibration steps

do_choose_calibrators=2
do_subtract=2
do_solve_dds4=2
do_smooth_dds4=2
do_slow_dds4=2
do_tec_inference=2
do_infer_screen=2
do_merge_slow=2

###
# imaging steps
do_image_smooth=0
do_image_dds4=0
do_image_smooth_slow=2
do_image_screen_slow=2
do_image_screen=0

###
# all args
L=(obs_num \
    archive_dir \
    root_working_dir \
    script_dir \
    region_file \
    mount_dirs \
    ncpu \
    do_image_smooth \
    do_image_dds4 \
    do_image_smooth_slow \
    do_image_screen_slow \
    do_image_screen \
    do_choose_calibrators \
    do_subtract \
    do_solve_dds4 \
    do_smooth_dds4 \
    do_slow_dds4 \
    do_tec_inference \
    do_infer_screen \
    do_merge_slow)

arg_parse_str="help"
for arg in ${L[@]}; do
    arg_parse_str=${arg_parse_str},${arg}:
done
echo $arg_parse_str

OPTS=$(getopt -o "h" --long ${arg_parse_str} -n "$progname" -- "$@")
if [ $? != 0 ] ; then echo "Error in command line arguments." >&2 ; usage; exit 1 ; fi
eval set -- "$OPTS"

while true; do
  # uncomment the next line to see how shift is working
  # echo "\$1:\"$1\" \$2:\"$2\""
  case "$1" in
    -h | --help ) usage; exit; ;;
    -- ) shift; break ;;
    * )
        for arg in ${L[@]}; do
            if [ "$1" == "--$arg" ]; then
                declare ${arg}="$2";
            fi
        done
        shift 2; break ;;
  esac
done

if [ -z "$obs_num" ]
then
    usage;
    exit;
fi

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
        --ncpu="$ncpu" \
        --block_size=20 \
        --deployment_type=directional \
        --no_subtract=False \
        --do_choose_calibrators="$do_choose_calibrators" \
        --do_subtract="$do_subtract" \
        --do_solve_dds4="$do_solve_dds4" \
        --do_smooth_dds4="$do_smooth_dds4" \
        --do_slow_dds4="$do_slow_dds4" \
        --do_tec_inference="$do_tec_inference" \
        --do_infer_screen="$do_infer_screen" \
        --do_merge_slow="$do_merge_slow" \
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
        --ncpu="$ncpu" \
        --block_size=10 \
        --deployment_type=directional \
        --do_image_smooth="$do_image_smooth" \
        --do_image_dds4="$do_image_dds4" \
        --do_image_smooth_slow="$do_image_smooth_slow" \
        --do_image_screen_slow="$do_image_screen_slow" \
        --do_image_screen="$do_image_screen" \
        --obs_num="$obs_num"
#        &>> "$log"
