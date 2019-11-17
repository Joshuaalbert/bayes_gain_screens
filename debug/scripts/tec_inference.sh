#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: $progname --script_dir --working_dir --data_dir --obs_num --ncpu --ref_dir --conda_env

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
ref_dir=0
ncpu=32

# use getopt and store the output into $OPTS
# note the use of -o for the short options, --long for the long name options
# and a : for any option that takes a parameter
OPTS=$(getopt -o "h" --long "help,conda_env:,script_dir:,obs_num:,data_dir:,working_dir:,ref_dir:,ncpu:" -n "$progname" -- "$@")
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
    --ref_dir ) ref_dir="$2"; shift 2 ;;
    --ncpu ) ncpu="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -z "$conda_env" ] || [ -z "$script_dir" ] || [ -z "$obs_num" ] || [ -z "$working_dir" ]
then
    usage;
    exit;
fi

source ~/.bashrc
source activate $conda_env
export PYTHONPATH=
cmd="python $script_dir/tec_inference.py --obs_num=$obs_num --data_dir=$data_dir --working_dir=$working_dir --ref_dir=$ref_dir --ncpu=$ncpu"
echo $cmd
eval $cmd
EC=$?
exit $?