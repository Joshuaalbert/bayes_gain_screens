import os
import glob
import pyregion
import numpy as np
import sys
import tables
import argparse


def make_masked_dico(region_mask, full_dico_model, masked_dico_model):
    cmd = 'MaskDicoModel.py --MaskName={region_mask} --InDicoModel={full_dico_model} --OutDicoModel={masked_dico_model} --InvertMask=1'.format(
        region_mask=region_mask, full_dico_model=full_dico_model, masked_dico_model=masked_dico_model)
    os.system(cmd)

def make_clustercat(reg_file, clustercat):
    regions = pyregion.open(reg_file)
    centers = np.zeros(len(regions[:]),
                       dtype=([('Name', 'S200'), ('ra', '<f8'), ('dec', '<f8'), ('SumI', '<f8'), ('Cluster', '<i8')]))

    print('Number of directions', len(regions[:]))

    for region_id, regions in enumerate(regions[:]):
        ra = np.pi * regions.coord_list[0] / 180.
        dec = np.pi * regions.coord_list[1] / 180.

        centers[region_id][0] = ''
        centers[region_id][1] = ra
        centers[region_id][2] = dec
        centers[region_id][3] = 0.
        centers[region_id][4] = region_id

    print("ClusterCat centers:\n{}".format(centers))
    np.save(clustercat, centers)


def solve(masked_dico_model, obs_num, clustercat, working_dir, data_dir, ncpu,sol_name):
    ddfcache = glob.glob(os.path.join(working_dir, '*.ddfcache'))
    for d in ddfcache:
        os.unlink(d)

    mslist = sorted(glob.glob(os.path.join(data_dir,'L{}*.ms'.format(obs_num))))
    if len(mslist) == 0:
        raise IOError("MS list  empty")

    if not os.path.isfile(clustercat):
        raise IOError("Clustercat doesn't exist {}".format(clustercat))

    solsdir = os.path.join(data_dir, 'SOLSDIR')
    out_sols = 'DDS4_full'

    for ms in mslist:
        cmd = 'kMS.py --MSName {ms} --SolverType KAFCA '.format(ms=ms)
        cmd = cmd + '--PolMode Scalar --BaseImageName image_full_ampphase_di_m.NS '
        cmd = cmd + '--dt 0.5 --NIterKF 6 --CovQ 0.1 --LambdaKF=0.5 --NCPU {ncpu}'.format(ncpu=ncpu)
        cmd = cmd + '--OutSolsName {out_sols} --NChanSols 1 --PowerSmooth=0.0 '.format(out_sols=sol_name)
        cmd = cmd + '--InCol DATA_SUB --Weighting Natural --UVMinMax=0.100000,5000.000000 '
        cmd = cmd + '--SolsDir={solsdir}  --BeamMode LOFAR --LOFARBeamMode=A --DDFCacheDir=. '.format(solsdir=solsdir)
        cmd = cmd + '--NodesFile {clustercat} --DicoModel {masked_dico_model}'.format(clustercat=clustercat, masked_dico_model=masked_dico_model)

        cmd = cmd.replace(' --', ' \\n--')
        with open(os.path.join(working_dir, 'instruct.sh'), 'w') as f:
            f.write(cmd)
        print(cmd)
        os.system(cmd)


def make_merged_h5parm(obs_num, sol_name, data_dir, working_dir):
    solsdir = os.path.join(data_dir, 'SOLSDIR')
    merged_h5parm = os.path.join(data_dir, 'L{}_{}_merged.h5'.format(obs_num, sol_name))
    sol_folders = sorted(glob.glob(os.path.join(solsdir, "L{}*.ms".format(obs_num))))
    if len(sol_folders) == 0:
        raise ValueError("Invalid obs num {}".format(obs_num))
    sols = []
    for f in sol_folders:
        sol = glob.glob(os.path.join(f, '*{}.sols.npz'.format(sol_name)))
        if len(sol) == 0:
            print("Can't find {} in {}".format(sol_name, f))
            continue
        sols.append(os.path.abspath(sol[0]))
    h5files = []
    for s in sols:
        cmd = 'killMS2H5parm.py --nofulljones {h5_file} {npz_file} '.format(npz_file=s,
                                                                            h5_file=s.replace('.npz', '.h5'))
        h5files.append(s.replace('.npz', '.h5'))
        print(cmd)
        os.system(cmd)

    cmd = 'H5parm_collector.py --outh5parm={merged_h5parm} [{h5files}]'.format(merged_h5parm=merged_h5parm,
                                                                               h5files=','.join(h5files))
    print(cmd)
    os.system(cmd)

    solsfile = os.path.join(working_dir, 'solslist.txt')
    merged_sol = os.path.join(data_dir, 'L{}_{}_merged.sols.npz'.format(obs_num, sol_name))
    with open(solsfile, 'w') as f:
        for s in sols:
            f.write("{}\n".format(s))
    os.system('MergeSols.py --SolsFilesIn={} --SolFileOut={}'.format(solsfile, merged_sol))


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument('--ncpu', help='Number of cpu to use, default=34', default=32, type=int)
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--region_file', help='Region file to use for new directions.',
                        default=None, type=str, required=True)

def main(region_file, obs_num, data_dir, working_dir, ncpu):
    clustercat = os.path.join(data_dir, 'subtract.ClusterCat.npy')
    region_mask = os.path.join(data_dir, 'cutoutmask.fits')
    full_dico_model = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.DicoModel')
    masked_dico_model = os.path.join(data_dir, 'image_full_ampphase_di_m.NS.masked.DicoModel')
    os.chdir(working_dir)
    make_clustercat(region_file, clustercat)
    make_masked_dico(region_mask, full_dico_model, masked_dico_model)
    solve(masked_dico_model,obs_num, clustercat, working_dir,data_dir, ncpu, 'DDS4_full')
    make_merged_h5parm(obs_num, 'DDS4_full', data_dir, working_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep soures inside box region, subtract everything else and create new ms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))