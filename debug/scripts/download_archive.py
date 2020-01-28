#!/usr/bin/env python
import subprocess
import os
import numpy as np
import argparse
import glob


def cmd_call(cmd):
    print("{}".format(cmd))
    exit_status = subprocess.call(cmd, shell=True)
    if exit_status:
        raise ValueError("Failed to  run: {}".format(cmd))

def get_solutions_timerange(sols):
    t = np.load(sols)['BeamTimes']
    return np.min(t), np.max(t)


def fixsymlinks(archive_dir, working_dir, obs_num):
    # Code from Tim for fixing symbolic links for DDS3_
    # dds3smoothed = glob.glob('SOLSDIR/*/*killMS.DDS3_full_smoothed*npz')
    print("Fixing symbolic links")
    dds3 = glob.glob(os.path.join(archive_dir, 'SOLSDIR/L{obs_num}*.ms/killMS.DDS3_full.sols.npz'.format(obs_num=obs_num)))
    for f in dds3:
        ms = os.path.basename(os.path.dirname(f))
        to_folder = os.path.join(working_dir, 'SOLSDIR', ms)
        try:
            os.makedirs(to_folder)
        except:
            pass
        for g in glob.glob(os.path.join(os.path.dirname(f), '*')):
            src = os.path.abspath(g)
            dst = os.path.join(to_folder, os.path.basename(g))
            if os.path.islink(dst):
                os.unlink(dst)
            print("Linking {} -> {}".format(src,dst))
            os.symlink(src,dst)
        start_time, _ = get_solutions_timerange(f)
        start_time = os.path.basename(
            glob.glob(os.path.join(archive_dir, 'DDS3_full_{}*_smoothed.npz'.format(int(start_time))))[0]).split('_')[2]
        src = os.path.join(archive_dir, 'DDS3_full_{}_smoothed.npz'.format(start_time))
        dst = os.path.join(to_folder, 'killMS.DDS3_full_smoothed.sols.npz')
        if os.path.islink(dst):
            os.unlink(dst)
        print("Linking {} -> {}".format(src, dst))
        os.symlink(src, dst)

        src = os.path.join(archive_dir, 'image_full_ampphase_di_m.NS.app.restored.fits')
        dst = os.path.join(working_dir, 'image_full_ampphase_di_m.NS.app.restored.fits')
        if os.path.islink(dst):
            os.unlink(dst)
        print("Linking {} -> {}".format(src, dst))
        os.symlink(src, dst)

def copy_archives(archive_dir, working_dir, obs_num, no_download):
    print("Copying archives.")
    archive_fullmask = os.path.join(archive_dir, 'image_full_ampphase_di_m.NS.mask01.fits')
    archive_indico = os.path.join(archive_dir, 'image_full_ampphase_di_m.NS.DicoModel')
    archive_clustercat = os.path.join(archive_dir, 'image_dirin_SSD_m.npy.ClusterCat.npy')
    fullmask = os.path.join(working_dir, os.path.basename(archive_fullmask))
    indico = os.path.join(working_dir, os.path.basename(archive_indico))
    clustercat = os.path.join(working_dir, 'image_dirin_SSD_m.npy.ClusterCat.npy')

    if no_download:
        cmd_call('mv {} {}'.format(archive_fullmask, fullmask))
        cmd_call('mv {} {}'.format(archive_indico, indico))
        cmd_call('mv {} {}'.format(archive_clustercat, clustercat))
    else:
        cmd_call('rsync -auvP {} {}'.format(archive_fullmask, fullmask))
        cmd_call('rsync -auvP {} {}'.format(archive_indico, indico))
        cmd_call('rsync -auvP {} {}'.format(archive_clustercat, clustercat))
    mslist = sorted(glob.glob(os.path.join(archive_dir, 'L{obs_num}*_SB*.ms.archive'.format(obs_num=obs_num))))
    print('Found archives files:\n{}'.format(mslist))
    outms = []
    for ms in mslist:
        outname = os.path.join(working_dir, os.path.basename(ms.rstrip('.archive')))
        if no_download:
            cmd_call('mv {}/ {}/'.format(ms, outname))
        else:
            cmd_call('rsync -auvP --delete {}/ {}/'.format(ms, outname))
        outms.append(outname)
    mslist_file = os.path.join(working_dir, 'mslist.txt')
    with open(mslist_file, 'w') as f:
        for ms in outms:
            f.write('{}\n'.format(ms))
    return mslist_file, outms, fullmask, indico, clustercat

def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--archive_dir', help='Where are the archives stored, may also be networked, e.g. <user>@<host>:<path>.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the subtract.',
                        default=None, type=str, required=True)
    parser.add_argument('--no_download', help='Whether to move instead of copy.',
                        default=False, type="bool", required=False)

def main(archive_dir, working_dir, obs_num, no_download):
    if no_download:
        if "SP_AUTH" in os.environ.keys():
            if os.environ['SP_AUTH'] != '1':
                raise ValueError("Trying to mv archive directory without authentication.")
        else:
            raise ValueError("Trying to mv archive directory without authentication.")
        print("Will use 'mv' instead of 'rsync'. Archive dir must be local then.")
    archive_dir = os.path.abspath(archive_dir)
    working_dir = os.path.abspath(working_dir)
    try:
        os.makedirs(working_dir)
    except:
        pass
    try:
        os.makedirs(os.path.join(working_dir, 'SOLSDIR'))
    except:
        pass
    os.chdir(working_dir)
    mslist_file, mslist, fullmask, indico, clustercat = copy_archives(archive_dir, working_dir, obs_num,no_download)
    if not os.path.isfile(fullmask):
        raise IOError("Missing mask {}".format(fullmask))
    if not os.path.isfile(indico):
        raise IOError("Missing dico model {}".format(indico))
    if not os.path.isfile(clustercat):
        raise IOError("Missing clustercat {}".format(clustercat))
    fixsymlinks(archive_dir, working_dir, obs_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the archive to root.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
