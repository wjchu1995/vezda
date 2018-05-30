# Copyright 2017-2018 Aaron C. Prunty
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#        
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==============================================================================

import numpy as np
from scipy.linalg import norm

def FundSol(pulseFun, recordingTimes, velocity, receiverPoints, samplingPoint):
    #RHS  right-hand side of the near-field equation in 2D
    #
    # Inputs:
    #   pulseFun: incident temporal pulse function (function handle)
    #   recordingTimes: recording time array (row or column vector of length 'Nt')
    #   timeShift: time shift parameter (positive constant).
    #   observationPoints: Nm-by-2 matrix, specifies the coordinates of the 
    #      observation points, where 'Nm' is the number of the points.
    #   samplingPoint: 1-by-2 vector specifying the current sampling point. 
    #                samplingPoint = [x(ix), y(iy)].
    #   incidentDirection: 1-by-2 vector, incident direction vector of dipole.
    #
    # Output:
    #   rhsFun: (Nt*Nm)-by-1 vector, specifies the monopole\dipole test function

    Nr = receiverPoints.shape[0] # number of receivers
    dim = receiverPoints.shape[1]
    
    # compute the distance between each receiver and the sampling point
    r = np.zeros(Nr)
    for i in range(Nr):
        r[i] = norm(receiverPoints[i,:] - samplingPoint) # |x - z|
    
    T, R = np.meshgrid(recordingTimes, r)
    
    retardedTime = T - R / velocity
    
    eps = np.finfo(float).eps     # about 2e-16 (so we never divide by zero)
    pulse = pulseFun(retardedTime)
    if dim == 2:
        sTR = np.lib.scimath.sqrt(T**2 - (R / velocity)**2)
        rhsFun = np.divide(pulse, 2 * np.pi * sTR + eps)
    elif dim == 3:
        rhsFun = np.divide(pulse, 4 * np.pi * velocity**2 * R + eps)
    
    #if sourceType == 'm':
    #    rhsFun = pulse./(2*pi*sTR) # monopole
    #else:
    #    # derivative of the pulse function
    #    tau = retardedTime;
    #    dpulse = (4*cos(4*tau)-3.2*(tau-3).*sin(4*tau)).*exp(-1.6*(tau-3).^2)
    #   xz = [observationPoints(:,1)-samplingPoint(1), ...
    #     observationPoints(:,2)-samplingPoint(2)] # x-z
    #    nuxz = incidentDirection(1)*xz(:,1)+incidentDirection(2)*xz(:,2) # nu.(x-z)
    #    NXXZ = repmat(nuxz,1,length(t))
    #    rhsFun = NXXZ.*(R.*pulse-sTR.*dpulse)./(2*pi*sTR.^3) # dipole
        
    rhsFun[retardedTime<=0] = 0    # causality
    rhsFun = np.real(rhsFun)        
    
    return rhsFun
