import numpy as np
import scipy.io as io

dataStructure = io.loadmat('starfish.mat')

receiverPoints = dataStructure['receivers']
sourcePoints = dataStructure['receivers']
scattererPoints = dataStructure['scatterer']
scatteredData = dataStructure['scatteredData']
recordingTimes = dataStructure['recordTimes']
recordingTimes = np.reshape(recordingTimes, (recordingTimes.shape[1],))

np.save('receiverPoints.npy', receiverPoints)
np.save('sourcePoints.npy', sourcePoints)
np.save('scattererPoints.npy', scattererPoints)
np.save('scatteredData.npy', scatteredData)
np.save('recordingTimes.npy', recordingTimes)
