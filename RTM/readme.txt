Efficient collapsed Gibbs sampling for the Relational Topic Model
(Version 1.0, last modified 11/12/2010).

This package implements the relational topic model of Chang & Blei.
The inference algorithm is collapsed Gibbs sampling.

Please see the little note "rtm.pdf" on the web site for the
sampling details.  The main function to run is rtmCGS.m.
Note that the RTMmex.c and mex_grad.c need to be compiled in Matlab first:

>> mex RTMmex.c
>> mex mex_grad.c

Example function call (please see function header for details on inputs):

>> rtmCGS('n','5','1','0','.1','0','2')

