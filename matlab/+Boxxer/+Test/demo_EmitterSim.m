imageSize = [256,512];
psfSigma = [1.1, 1.1];
nParticles = 100;
nFrames = 10;
import('Boxxer.EmitterSim')
es = EmitterSim(imageSize, psfSigma);
[im2d, loc2d] = es.simulate2DImage(nParticles, nFrames);
