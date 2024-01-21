import os
import sys
import random
from PandAna import *

## Version 7: Elkarghli 11052020 ##
## Data Quality Cuts (return tables arr) ##

# Veto (no-go) region
kVeto = Cut(lambda tables: tables['rec.sel.veto']['keep'] == 1)
# Vertex mst be contained within the ND detector
kVtxCont = Cut(lambda tables: tables['rec.mc.nu']['isvtxcont'] == 1)
# Vertex must be present (for reco vtx)
#kVtx  = Cut(lambda tables: tables['rec.vtx']['nelastic'] > 0)
kVtx  = Cut(lambda tables: tables['rec.vtx.elastic']['IsValid'])
# Prongs must be present
kPng  = Cut(lambda tables: tables['rec.vtx.elastic.fuzzyk']['npng'] > 0)
# NuE Cosmic Rejection filter
kFEB  = Cut(lambda tables: tables['rec.sel.nuecosrej']['hitsperplane'] < 8)

## Containment Cut - interactions not on edges

def kContain(tables):
    df = tables['rec.sel.nuecosrej']
    return \
        (df['distallpngtop'] > 30) & \
        (df['distallpngbottom'] > 30) & \
        (df['distallpngeast'] > 30) & \
        (df['distallpngwest'] > 30) & \
        (df['distallpngfront'] > 30) & \
        (df['distallpngback'] > 30)
kContain = Cut(kContain)

## Particle Data Group (PDG): Electron (12/-12) or Muon (14/-14) Neutrino & Antineutrino
## Charged Current Interactions (ISCC): COMMENTED OUT
def kNueOrNumu(tables):
    pdg = tables['rec.mc.nu']['pdg']
    cc = tables['rec.mc.nu']['iscc']

    return ((pdg==12) | (pdg==14) | (pdg==-12) | (pdg==-14)) #& (cc==1)
kNueOrNumu = Cut(kNueOrNumu)

## Ensure the values of the x,y,z-vertices are not outside the detector limits
def kVtxBound(tables):
    xMax = tables['rec.mc.nu']['vtx.x']
    yMax = tables['rec.mc.nu']['vtx.y']
    zMax = tables['rec.mc.nu']['vtx.z']

    return ((abs(xMax)<=195.) & (abs(yMax)<=195.) & (zMax<=1400.) & (zMax>=2.))
kVtxBound = Cut(kVtxBound)

## Extract the neutrino mode (0: QE, 1: RES, 2: DIS, 3: Coh, 10: MEC)
def kMode(tables):
    return tables['rec.mc.nu']['mode']
kMode = Var(kMode)

## Extract CC/NC Boolean
def kisCC(tables):
    return tables['rec.mc.nu']['iscc']
kisCC = Var(kisCC)

## Extract the PDG with sign
def kSign(tables):
    return tables['rec.mc.nu']['pdg']
kSign = Var(kSign)

## Extract Neutrino Energy
def kEnergy(tables):
    return tables['rec.mc.nu']['E']

## Extract True Monte-Carlo Vertices
def kVtxx(tables):
    return tables['rec.mc.nu']['vtx.x']
kVtxx = Var(kVtxx)

def kVtxy(tables):
    return tables['rec.mc.nu']['vtx.y']
kVtxy = Var(kVtxy)

def kVtxz(tables):
    return tables['rec.mc.nu']['vtx.z']
kVtxz = Var(kVtxz)

## Extract Reco-Elastc Vertices
def kRecox(tables):
    return tables['rec.vtx.elastic']['vtx.x']
kRecox = Var(kRecox)

def kRecoy(tables):
    return tables['rec.vtx.elastic']['vtx.y']
kRecoy = Var(kRecoy)

def kRecoz(tables):
    return tables['rec.vtx.elastic']['vtx.z']
kRecoz = Var(kRecoz)

## Extract Convolutional Visual Network Pixel Map
def kMap(tables):
    return tables['rec.training.cvnmaps']['cvnmap']

## Extract Cell-Plane References of PM
def kFirstPlane(tables):
    return tables['rec.training.cvnmaps']['firstplane']
kFirstPlane = Var(kFirstPlane)

def kLastCellX(tables):
    return tables['rec.training.cvnmaps']['lastcellx']
kLastCellX = Var(kLastCellX)

def kLastCellY(tables):
    return tables['rec.training.cvnmaps']['lastcelly']
kLastCellY = Var(kLastCellY)


if __name__ == '__main__':
    # Terminal Arguments: input directory [1] output directory [2]
    indir = sys.argv[1]
    outdir = sys.argv[2]
    print('Process h5 files in '+indir+' to training files in '+outdir)
    files = [f for f in os.listdir(indir) if 'h5caf.h5' in f]
    files = random.sample(files, len(files))
    print('There are '+str(len(files))+' files.')

    # Define Cut Aggregate
    kCut = kVeto & kVtxCont & kVtx & kPng & kFEB & kContain & kNueOrNumu & kVtxBound

    # One file at a time to avoid problems with loading a bunch of pixel maps in memory
    for i,f in enumerate(files):
        # Definte the output name and don't recreate it
        outname = '{0}_TrainData{1}'.format(f[:-9], f[-9:])
        if os.path.exists(os.path.join(outdir,outname)):
            continue

        # Make a loader and the two spectra to fill
        tables = loader([os.path.join(indir,f)])

        specMap    = spectrum(tables, kCut, kMap)
        specSign   = spectrum(tables, kCut, kSign)
        specMode   = spectrum(tables, kCut, kMode)
        specisCC   = spectrum(tables, kCut, kisCC)
        specEnergy = spectrum(tables, kCut, kEnergy)
        specVtxx   = spectrum(tables, kCut, kVtxx)
        specVtxy   = spectrum(tables, kCut, kVtxy)
        specVtxz   = spectrum(tables, kCut, kVtxz)

        specRecox  = spectrum(tables, kCut, kRecox)
        specRecoy  = spectrum(tables, kCut, kRecoy)
        specRecoz  = spectrum(tables, kCut, kRecoz)

        specFirstPlane  = spectrum(tables, kCut, kFirstPlane)
        specLastCellX   = spectrum(tables, kCut, kLastCellX)
        specLastCellY   = spectrum(tables, kCut, kLastCellY)

        # Process
        tables.Go()

        # Return error if NaN fields
        if specMap.entries()==0:
            print(str(i)+': File '+f+' is empty. Modify selection criteria.')
            continue

        # Concatenate the dataframes to line up label and map
        # join='inner' ensures there is both a label and a map for the slice
        df = pd.concat([specMap.df(),specSign.df(),specMode.df(),specisCC.df(),specEnergy.df(),specVtxx.df(),specVtxy.df(),specVtxz.df(),specRecox.df(),specRecoy.df(),specRecoz.df(),specFirstPlane.df(),specLastCellX.df(),specLastCellY.df()], axis=1, join='inner').reset_index()
        
        # Save in an h5 with new dataset keys
        hf = h5py.File(os.path.join(outdir,outname),'w')

        hf.create_dataset('firstplane', data=df['firstplane'],              compression='gzip')
        hf.create_dataset('lastcellx',  data=df['lastcellx'],               compression='gzip')
        hf.create_dataset('lastcelly',  data=df['lastcelly'],               compression='gzip')

        hf.create_dataset('TrueRecoVtxX',  data=df['vtx.x'],                compression='gzip')
        hf.create_dataset('TrueRecoVtxY',  data=df['vtx.y'],                compression='gzip')
        hf.create_dataset('TrueRecoVtxZ',  data=df['vtx.z'],                compression='gzip')
      
        hf.create_dataset('run',       data=df['run'],                      compression='gzip')
        hf.create_dataset('subrun',    data=df['subrun'],                   compression='gzip')
        hf.create_dataset('cycle',     data=df['cycle'],                    compression='gzip')
        hf.create_dataset('event',     data=df['evt'],                      compression='gzip')
        hf.create_dataset('slice',     data=df['subevt'],                   compression='gzip') 
        hf.create_dataset('PDG',       data=df['pdg'],                      compression='gzip')
        hf.create_dataset('Mode',      data=df['mode'],                     compression='gzip')
        hf.create_dataset('isCC',      data=df['iscc'],                     compression='gzip')
        hf.create_dataset('E',         data=df['E'],                        compression='gzip')

        hf.create_dataset('cvnmap',    data=np.stack(df['cvnmap']),         compression='gzip')

        hf.close()
