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
obs_num=
archive_dir=
remote_archive=
root_working_dir=
script_dir=
region_file=None
bind_dirs=
ncpu=$(grep -c ^processor /proc/cpuinfo)
conda_env=bayes_gain_screens_py
force_conda=
auto_resume=2
no_download=False
mock_run=
retry_task_on_fail=0

###
# calibration steps

do_download_archive=2
do_choose_calibrators=2
do_subtract=2
do_subtract_outside_pb=2
do_solve_dds4=2
do_neural_gain_flagger=2
do_slow_solve_dds4=2
do_tec_inference_and_smooth=2
do_infer_screen=2
do_merge_slow=2
do_flag_visibilities=2

###
# imaging steps
do_image_subtract_dirty=0
do_image_smooth=0
do_image_subtract_dds4=0
do_image_dds4=0
do_image_smooth_slow=0
do_image_smooth_slow_restricted=0
do_image_screen_slow=0
do_image_screen_slow_restricted=2
do_image_screen=0


###
# all args
L=(obs_num \
    archive_dir \
    remote_archive \
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
    do_neural_gain_flagger \
    do_slow_solve_dds4 \
    do_tec_inference_and_smooth \
    do_infer_screen \
    do_merge_slow \
    do_flag_visibilities \
    simg_dir \
    conda_env \
    force_conda \
    no_download \
    auto_resume \
    mock_run \
    retry_task_on_fail \
    const_smooth_window)

arg_parse_str="help"
for arg in "${L[@]}"; do
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
  for arg in "${L[@]}"; do
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

if [ -z "$archive_dir" ]
then
    usage;
    exit;
fi



if [ -z "$force_conda" ]; then
bayes_gain_screens_simg="$simg_dir"/bayes_gain_screens.simg
else
bayes_gain_screens_simg=None
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate $conda_env

fi

if [ -z "$remote_archive" ]; then
    echo "Archive assumed to be local"
else
  if [ -d "$archive_dir" ]; then
    echo "$archive_dir seems to exist already. Updating the contents with rsync."
  fi
  mkdir -p $archive_dir
  rsync -avuP \
      "$remote_archive"/L*.archive \
      "$remote_archive"/SOLSDIR \
      "$remote_archive"/*.app.restored.fits \
      "$remote_archive"/image_full_ampphase_di_m.NS.mask01.fits \
      "$remote_archive"/image_full_ampphase_di_m.NS.DicoModel \
      "$remote_archive"/DDS3_full_*smoothed.npz \
      "$remote_archive"/DDS3_full_slow_*.npz \
      "$remote_archive"/image_dirin_SSD_m.npy.ClusterCat.npy \
      "$archive_dir"/
  chmod -R u=rwxt "$archive_dir"
  chmod -R go=rx "$archive_dir"
fi

#source ~/.bashrc
#ddf_singularity

singularity exec -B /tmp,/dev/shm "$simg_dir"/lofar_sksp_ddf.simg CleanSHM.py

cmd="python $HOME/git/bayes_gain_screens/bin/gain_screens_pipeline.py \
        --archive_dir=$archive_dir \
        --root_working_dir=$root_working_dir \
        --script_dir=$script_dir \
        --region_file=$region_file \
        --auto_resume=$auto_resume \
        --ncpu=$ncpu \
        --deployment_type=directional \
        --retry_task_on_fail=$retry_task_on_fail \
        --no_download=$no_download \
        --do_download_archive=$do_download_archive \
        --do_choose_calibrators=$do_choose_calibrators \
        --do_subtract=$do_subtract \
        --do_subtract_outside_pb=$do_subtract_outside_pb \
        --do_solve_dds4=$do_solve_dds4 \
        --do_neural_gain_flagger=$do_neural_gain_flagger \
        --do_slow_solve_dds4=$do_slow_solve_dds4 \
        --do_tec_inference_and_smooth=$do_tec_inference_and_smooth \
        --do_infer_screen=$do_infer_screen \
        --do_merge_slow=$do_merge_slow \
        --do_flag_visibilities=$do_flag_visibilities \
        --do_image_smooth=$do_image_smooth \
        --do_image_subtract_dds4=$do_image_subtract_dds4 \
        --do_image_dds4=$do_image_dds4 \
        --do_image_smooth_slow=$do_image_smooth_slow \
        --do_image_smooth_slow_restricted=$do_image_smooth_slow_restricted \
        --do_image_screen_slow=$do_image_screen_slow \
        --do_image_screen_slow_restricted=$do_image_screen_slow_restricted \
        --do_image_screen=$do_image_screen \
        --obs_num=$obs_num \
        --bind_dirs=$bind_dirs \
        --lofar_sksp_simg=$simg_dir/lofar_sksp_ddf.simg \
        --lofar_gain_screens_simg=$simg_dir/lofar_sksp_ddf_gainscreens_premerge.simg \
        --bayes_gain_screens_simg=$bayes_gain_screens_simg \
        --bayes_gain_screens_conda_env=$conda_env"

if [ -z "$mock_run" ]; then
  eval $cmd
else
  echo $cmd
  echo Mock run, exitting before run.
fi