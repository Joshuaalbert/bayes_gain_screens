#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $progname --script_dir --working_dir --data_dir --obs_num --conda_env --ncpu --ref_dir --deployment_type
   --block_size --ref_image_fits

   optional arguments:
     -h, --help           show this help message and exit

HEREDOC
}

# initialize variables
progname=$(basename $0)
conda_env="tf_py"
script_dir=
obs_num=
data_dir=
working_dir=
ncpu=32
ref_dir=0
deployment_type="directional"
block_size=10
ref_image_fits=

# use getopt and store the output into $OPTS
# note the use of -o for the short options, --long for the long name options
# and a : for any option that takes a parameter
OPTS=$(getopt -o "h" --long "help,conda_env:,script_dir:,obs_num:,data_dir:,working_dir:,ncpu:,ref_dir:,deployment_type:,block_size:,ref_image_fits:" -n "$progname" -- "$@")
if [ $? != 0 ] ; then echo "Error in command line arguments." >&2 ; usage; exit 1 ; fi
eval set -- "$OPTS"

while true; do
  # uncomment the next line to see how shift is working
  # echo "\$1:\"$1\" \$2:\"$2\""
  case "$1" in
    -h | --help ) usage; exit; ;;
    --conda_env ) conda_env="$2"; shift 2 ;;
    --script_dir ) script_dir="$2"; shift 2 ;;
    --working_dir ) working_dir="$2"; shift 2 ;;
    --data_dir ) data_dir="$2"; shift 2 ;;
    --obs_num ) obs_num="$2"; shift 2 ;;
    --ncpu ) ncpu="$2"; shift 2 ;;
    --ref_dir ) ref_dir="$2"; shift 2 ;;
    --deployment_type ) deployment_type="$2"; shift 2 ;;
    --block_size ) block_size="$2"; shift 2 ;;
    --ref_image_fits ) ref_image_fits="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -z "$conda_env" ] || [ -z "$script_dir" ] || [ -z "$obs_num" ] || [ -z "$working_dir" ] || [ -z "$data_dir" ] \
|| [ -z "$ncpu" ] || [ -z "$ref_dir" ] || [ -z "$ref_image_fits" ]
then
    usage;
    exit;
fi

source ~/.bashrc
source activate $conda_env
export PYTHONPATH=
cmd="python $script_dir/infer_screen.py --obs_num=$obs_num --data_dir=$data_dir --working_dir=$working_dir \
--ncpu=$ncpu --ref_dir=$ref_dir --deployment_type=$deployment_type --block_size=$block_size \
--ref_image_fits=$ref_image_fits"
echo $cmd
eval $cmd
