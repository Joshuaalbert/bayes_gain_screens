import os
import glob
import pyregion
import numpy as np
import sys
import tables

#cmake ../ -DCMAKE_PREFIX_PATH="/net/rijn/data2/rvweeren/software/idggpu_aug7_2019/opt;/net/lofar1/data1/rvweeren/software/LOFARBeam/opt;/net/lofar1/data1/oonk/rh7_lof_aug2017_trunk/casacore;/net/lofar1/data1/oonk/rh7_lof_aug2017_trunk/cfitsio;/software/rhel7"

#Running: kMS.py --MSName L719874_SB001_uv_12DF5BEDAt_166MHz.pre-cal.ms --SolverType KAFCA --PolMode Scalar \
#--BaseImageName image_full_ampphase_di_m --dt 43.630000 --NIterKF 6 --CovQ 0.100000 --LambdaKF=0.500000 --NCPU 48 \
#--OutSolsName DDS3_full_slow --InCol DATA_DI_CORRECTED --Weighting Natural --UVMinMax=0.500000,1000.000000 --SolsDir=SOLSDIR \
#--PreApplySols=[DDS3_full_smoothed] --NChanSols 1 --BeamMode LOFAR --LOFARBeamMode=A --DDFCacheDir=. --NodesFile image_dirin_SSD_m.npy.ClusterCat.npy --DicoModel image_full_ampphase_di_m_masked.DicoModel

#ClipCal.py --MSName L719874_SB001_uv_12DF5BEDAt_164MHz.pre-cal.ms --ColName DATA_DI_CORRECTED
#MergeSols.py --SolsFilesIn=solslist_5066047195.00556.txt --SolFileOut=DDS3_full_slow_5066047195.00556_merged.npz --SigmaFilterOutliers 5.000000

def fillkMSsolsfromH5():
    kms = np.load('../DDS4_full_merged.npz')
    H =tables.open_file('/net/rijn/data2/rvweeren/P126+65_recall/ClockTEC/P126+65_full_compact_ampphasesmoothed.h5', mode='r')
    phase   = H.root.sol000.phase000.val[:].T
    amp     = H.root.sol000.amplitude000.val[:].T
    H.close()
    sols = np.copy(kms['Sols'])
    
    print phase[:,:,:,:,0].shape
    print kms['Sols']['G'][:,:,:,:,0,0].shape

    assert phase[:,:,:,:,0].shape == kms['Sols']['G'][:,:,:,:,0,0].shape   
    
    #kms['Sols']['G'][:,:,:,:,0,0] = amp[:,:,:,:,0]*np.cos(phase[:,:,:,:,0]) + 1j * amp[:,:,:,:,0]*np.sin(phase[:,:,:,:,0]) # XX
    #kms['Sols']['G'][:,:,:,:,1,1] = amp[:,:,:,:,3]*np.cos(phase[:,:,:,:,3]) + 1j * amp[:,:,:,:,3]*np.sin(phase[:,:,:,:,3]) # YY

    sols['G'][:,:,:,:,0,0] = amp[:,:,:,:,0]*np.cos(phase[:,:,:,:,0]) + 1j * amp[:,:,:,:,0]*np.sin(phase[:,:,:,:,0]) # XX
    sols['G'][:,:,:,:,1,1] = amp[:,:,:,:,3]*np.cos(phase[:,:,:,:,3]) + 1j * amp[:,:,:,:,3]*np.sin(phase[:,:,:,:,3]) # YY

    
    xx = amp[:,:,:,:,0]*np.cos(phase[:,:,:,:,0]) + 1j * amp[:,:,:,:,0]*np.sin(phase[:,:,:,:,0]) # XX
    yy = amp[:,:,:,:,3]*np.cos(phase[:,:,:,:,3]) + 1j * amp[:,:,:,:,3]*np.sin(phase[:,:,:,:,3]) # YY
    
    print  sols['G'][:,:,:,:,0,0] - xx
    print  sols['G'][:,:,:,:,1,1] - yy
    
    # H5 npol, ndir, nant, nfreq, ntime
    
    #kMS = 960, 24, 62, 36, 2, 2
    # print kMS.keys()
    #['ModelName', 'MaskedSols', 'FreqDomains', 'StationNames', 'BeamTimes', 'SourceCatSub', 'ClusterCat', 'MSName', 'Sols', 'SkyModel']


    
    #np.savez('../DDS4_full_smoothedReinoout.npz', ModelName=kms['ModelName'], MaskedSols=kms['MaskedSols'], \
    #         FreqDomains=kms['FreqDomains'], StationNames=kms['StationNames'], BeamTimes=kms['BeamTimes'], \
    #         SourceCatSub=kms['SourceCatSub'], ClusterCat=kms['ClusterCat'], MSName=kms['MSName'], Sols=kms['Sols'], \
    #         SkyModel=kms['SkyModel'])

    np.savez('../DDS4_full_smoothedReinoout_fixed.npz', ModelName=kms['ModelName'], MaskedSols=kms['MaskedSols'], \
             FreqDomains=kms['FreqDomains'], StationNames=kms['StationNames'], BeamTimes=kms['BeamTimes'], \
             SourceCatSub=kms['SourceCatSub'], ClusterCat=kms['ClusterCat'], MSName=kms['MSName'], Sols=sols, \
             SkyModel=kms['SkyModel'])
    
    
    return


def kMSslow():
    mslist = sorted(glob.glob('L*.ms'))
    #mslist = mslist[5::]
    #print mslist
    #sys.exit()
    #dicomodel   = 'image_full_ampphase_di_m.NS.36dir.DicoModel' 
    dicomodel   = 'DR2_P126+65_full_compact_recal.DicoModel'
    clustercat  = 'compact.ClusterCat.npy'
    outsolsname = 'DDS4_full_slow_Josh_fixed'
    preapplysols= 'DDS4_full_smoothedReinoout_fixed'    
    for ms in mslist:

      cmd = 'kMS.py --MSName '+ ms + ' ' + '--SolverType KAFCA '
      cmd = cmd + '--PolMode Scalar --BaseImageName image_full_ampphase_di_m.NS ' 
      cmd = cmd + '--dt 43.630000 --NIterKF 6 --CovQ 0.1 --LambdaKF=0.5 --NCPU 48 ' 
      cmd = cmd + '--OutSolsName ' + outsolsname + ' --NChanSols 1 --PowerSmooth=0.0 '
      cmd = cmd + '--PreApplySols=['+preapplysols+ ']' + ' '
      cmd = cmd + '--InCol DATA_SUB --Weighting Natural --UVMinMax=0.5,5000.000000 ' 
      cmd = cmd + '--SolsDir=SOLSDIR  --BeamMode LOFAR --LOFARBeamMode=A --DDFCacheDir=. ' 
      cmd = cmd + '--NodesFile ' + clustercat + ' --DicoModel ' + dicomodel
    
      #if ms != 'cP126+65BEAM_1_chan0-10.ms' and ms != 'cP126+65BEAM_1_chan10-20.ms' : 
      print cmd
      os.system(cmd)



def MakeDico():
    
    cmd = 'MaskDicoModel.py --MaskName=cutoutmask.fits --InDicoModel=image_full_ampphase_di_m.NS.DicoModel --OutDicoModel=image_full_ampphase_di_m.NS.masked.DicoModel --InvertMask=1'
    os.system(cmd)

def kMS(dicomodel = "image_full_ampphase_di_m.NS.masked.DicoModel", obs_num=None, clustercat=None):
    
    os.system('rm -rf *.ddfcache')

    #mslist = sorted(glob.glob('cP126+65BEAM_1_chan*-*.ms'))
    if obs_num is None:
        mslist = sorted(glob.glob('L*.ms'))
    else:
        mslist = sorted(glob.glob('L{}*.ms'.format(obs_num)))
    if len(mslist)==0:
        raise ValueError("MS list  empty")

    if clustercat is None:
        clustercat = glob.glob("*ClusterCat.npy")
        if len(clustercat)==0:
            raise ValueError("No Cluster Cat passed or found")
        clustercat = clustercat[0]
    print("Using cluster cat {}".format(clustercat))

    outsolsname = 'DDS4_full'

    for ms in mslist:

      cmd = 'kMS.py --MSName '+ ms + ' ' + '--SolverType KAFCA '
      cmd = cmd + '--PolMode Scalar --BaseImageName image_full_ampphase_di_m.NS ' 
      cmd = cmd + '--dt 0.5 --NIterKF 6 --CovQ 0.1 --LambdaKF=0.5 --NCPU 32 ' 
      cmd = cmd + '--OutSolsName ' + outsolsname + ' --NChanSols 1 --PowerSmooth=0.0 ' 
      cmd = cmd + '--InCol DATA_SUB --Weighting Natural --UVMinMax=0.100000,5000.000000 ' 
      cmd = cmd + '--SolsDir=SOLSDIR  --BeamMode LOFAR --LOFARBeamMode=A --DDFCacheDir=. ' 
      cmd = cmd + '--NodesFile ' + clustercat + ' --DicoModel ' + dicomodel
    
      print cmd
      os.system(cmd)

def ClipCal():

     mslist = sorted(glob.glob('L*.ms'))  
     
     for ms in mslist:
     
       cmd = 'ClipCal.py --MSName ' + ms + ' --ColName DATA_SUB '
       print cmd
       os.system(cmd)
       
def MergeSols(solsname):

    #solsname = 'DDS5_full'
    #solsname = 'DDS4_full_slow'
    path     = '/net/rijn/data2/rvweeren/P126+65_recall/SOLSDIR/L562061*.pre-cal.ms/'
    path     = '/net/bovenrijn/data1/rvweeren/P126+65_recall_slowsolve/SOLSDIR/L562061*.pre-cal.ms/'
    path     = '/net/voorrijn/data2/rvweeren/P126+65_recall_slowsolve/SOLSDIR/L562061*.pre-cal.ms/'
    path     = '/net/krommerijn/data2/rvweeren/P126+65_recall_slowsolve/SOLSDIR/L562061*.pre-cal.ms/'
    cmd = 'ls -1 ' + path + '/killMS.' + solsname + '.sols.npz > solslist.txt'
    os.system(cmd)
    
    cmd = 'MergeSols.py --SolsFilesIn=solslist.txt --SolFileOut='+ solsname + '_merged.npz'
    print cmd
    os.system(cmd)

def SmoothSols():
    solsname = 'DDS5_full'
    cmd = 'SmoothSols.py --SolsFileIn=' + solsname + '_merged.npz --SolsFileOut='+ solsname + '_smoothed.npz --InterpMode=TEC,PolyAmp'    
    print cmd
    os.system(cmd)  
    
def makeClusterCat(reg_file):
    
    
    regions = pyregion.open(reg_file)    
    #dirs = np.load('image_dirin_SSD_m.npy.ClusterCat.npy')
    

    centers = np.zeros(len(regions[:]), dtype=([('Name', 'S200'), ('ra', '<f8'), ('dec', '<f8'), ('SumI', '<f8'), ('Cluster', '<i8')]))  
 
    
    print 'Number of directions', len(regions[:]) 
    
    for region_id,regions in enumerate(regions[:]):
        
      #print region_id
      ra  = np.pi*regions.coord_list[0]/180.
      dec = np.pi*regions.coord_list[1]/180.

      centers[region_id][0] = ''
      centers[region_id][1] = ra
      centers[region_id][2] = dec
      centers[region_id][3] = 0. 
      centers[region_id][4] = region_id 

      #print centers[region_id]
      #sys.exit()
    
    print centers
    np.save(reg_file.replace('.reg', '.ClusterCat.npy'),centers)

#MakeDico()
#SmoothSols() 
#fillkMSsolsfromH5()    
#kMSslow()   
#ClipCal()    
#MergeSols()
#kMS()
#makeClusterCat('/home/albert/store/lockman/LHdeepbright.reg')
kMS(dicomodel = "image_full_ampphase_di_m.NS.masked.DicoModel", 
        obs_num=667218,clustercat='LHdeepbright.ClusterCat.npy')
