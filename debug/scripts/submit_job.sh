#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $progname --obs_num [--archive_dir --root_working_dir --script_dir --region_file --simg_dir --bind_dirs --ncpu]

   optional arguments:
     -h, --help           show this help message and exit

HEREDOC
}

# initialize variables and defaults
progname=$(basename $0)
# wheres
simg_dir=${HOME}/store
obs_num=562061
archive_dir=${HOME}/store/P126+65
root_working_dir=${HOME}/store/root
script_dir=${HOME}/store/scripts
region_file=None
bind_dirs=/beegfs/lofar
ncpu=$(grep -c ^processor /proc/cpuinfo)
conda_env=tf_py
force_conda=
auto_resume=True
no_download=False

###
# calibration steps

do_download_archive=2
do_choose_calibrators=2
do_subtract=2
do_subtract_outside_pb=2
do_solve_dds4=2
do_smooth_dds4=2
do_slow_solve_dds4=2
do_tec_inference=2
do_infer_screen=2
do_merge_slow=2

###
# imaging steps
do_image_subtract_dirty=0
do_image_smooth=0
do_image_subtract_dds4=0
do_image_dds4=0
do_image_smooth_slow=2
do_image_smooth_slow_restricted=0
do_image_screen_slow=2
do_image_screen_slow_restricted=0
do_image_screen=0

###
# all args
L=(obs_num \
    archive_dir \
    root_working_dir \
    script_dir \
    region_file \
    bind_dirs \
    ncpu \
    do_image_smooth \
    do_image_subtract_dds4 \
    do_image_dds4 \
    do_image_smooth_slow \
    do_image_smooth_slow_restricted \
    do_image_screen_slow \
    do_image_screen_slow_restricted \
    do_image_screen \
    do_download_archive \
    do_choose_calibrators \
    do_subtract \
    do_subtract_outside_pb \
    do_solve_dds4 \
    do_smooth_dds4 \
    do_slow_solve_dds4 \
    do_tec_inference \
    do_infer_screen \
    do_merge_slow \
    simg_dir \
    conda_env \
    force_conda \
    no_download \
    auto_resume)

arg_parse_str="help"
for arg in ${L[@]}; do
    arg_parse_str=${arg_parse_str},${arg}:
done
#echo $arg_parse_str

OPTS=$(getopt -o "h" --long ${arg_parse_str} -n "$progname" -- "$@")
if [ $? != 0 ] ; then echo "Error in command line arguments." >&2 ; usage; exit 1 ; fi
eval set -- "$OPTS"

while true; do
  # uncomment the next line to see how shift is working
  # echo "\$1:\"$1\" \$2:\"$2\""
  case "$1" in
    -h | --help ) usage; exit; ;;
    -- ) shift; break ;;
  esac
  found=
  for arg in ${L[@]}; do
    if [ "$1" == "--$arg" ]; then
        declare ${arg}="$2";
        shift 2;
        found=1
        break
    fi
  done
  if [ -z "$found" ]; then
    break
  fi
done

if [ -z "$obs_num" ]
then
    usage;
    exit;
fi

if [ -z "$force_conda" ]; then
bayes_gain_screens_simg="$simg_dir"/bayes_gain_screens.simg
else
bayes_gain_screens_simg=None
fi

#source ~/.bashrc
#ddf_singularity

singularity exec -B /tmp,/dev/shm "$simg_dir"/lofar_sksp_ddf.simg CleanSHM.py

python "$script_dir"/pipeline.py \
        --archive_dir="$archive_dir" \
        --root_working_dir="$root_working_dir" \
        --script_dir="$script_dir" \
        --region_file="$region_file" \
        --auto_resume="$auto_resume" \
        --ref_dir=0 \
        --ncpu="$ncpu" \
        --block_size=20 \
        --deployment_type=directional \
        --no_download="$no_download" \
        --do_download_archive="$do_download_archive" \
        --do_choose_calibrators="$do_choose_calibrators" \
        --do_subtract="$do_subtract" \
        --do_subtract_outside_pb="$do_subtract_outside_pb" \
        --do_solve_dds4="$do_solve_dds4" \
        --do_smooth_dds4="$do_smooth_dds4" \
        --do_slow_solve_dds4="$do_slow_solve_dds4" \
        --do_tec_inference="$do_tec_inference" \
        --do_infer_screen="$do_infer_screen" \
        --do_merge_slow="$do_merge_slow" \
        --do_image_smooth="$do_image_smooth" \
        --do_image_subtract_dds4="$image_do_image_subtract_dds4" \
        --do_image_dds4="$do_image_dds4" \
        --do_image_smooth_slow="$do_image_smooth_slow" \
        --do_image_smooth_slow_restricted="$do_image_smooth_slow_restricted" \
        --do_image_screen_slow="$do_image_screen_slow" \
        --do_image_screen_slow_restricted="$do_image_screen_slow_restricted" \
        --do_image_screen="$do_image_screen" \
        --obs_num="$obs_num" \
        --bind_dirs="$bind_dirs" \
        --lofar_sksp_simg="$simg_dir"/lofar_sksp_ddf.simg \
        --lofar_gain_screens_simg="$simg_dir"/lofar_sksp_ddf_gainscreens_premerge.simg \
        --bayes_gain_screens_simg="$bayes_gain_screens_simg" \
        --bayes_gain_screens_conda_env="$conda_env"

