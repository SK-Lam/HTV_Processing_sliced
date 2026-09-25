Pipeline + init : the integrated code to run each module
Config: all constants and setups are input here
Common: the processing and masking shared by displacement & tilt (Step 1-4)
Displacement & tilt: separated processing of 
Phase: Step 6    phi_tilt = 2 pi f0 . r
            Step 7    B = A exp(-i phi_tilt) = (1/2) I_env V exp(i phi_0)
            Step 10   phi_dis = arg( SUM a(x,y) B(x,y) )        Eq. (18)
            Step 11   dz = (lambda / 4 pi) phi_dis              Eq. (19)

 Fringes: Step 5  theta = (lambda / 2p) * f0                      Eq. (9)
Htv.io, Report : functions to setup input and output of files Spectra : this is the welch method chunk (downstream of the protocol) with hann window applied  

Steps based on the protocol
