import numpy as np
import Profile_Likelihood as PL
import Custom_Estimation_Routines as CER
import Error_Model as EM
import SCB_estimate as SCBe

def GPL_estimation(varying_parameters,Data,t,Reference_parameters,References_likelihoods,Bounds,verbose_success=True,verbose_error=False):
    """Computes a generalized version of profile likelihood, where any number of parameters are allowed to vary
    Arguments:
    ----------
    varying_parameters (list): list of the indexes of the parameters that are allowed to vary (which thus need to be estimated)
    Data (array): data to be fitted
    t (array): time
    Reference_parameters (array): reference values of the parameters (in the case some parameters won't vary)
    References_likelihood=(L1, L2): likelihoods of the reference parameters in the two steps of the estimation
    Bounds (list of 2-uples): bounds of the parameter values
    verbose_success (boolean): whether or not to display messages when estimation succeeds
    verbose_error (boolean): whether or not to display messages when estimation failed

    Returns:
    --------
    dict, a dictionary with two keys:
        'parameters': the estimated parameter values (including the parameters that don't vary)
        'error': the value of -log(Likelihood) for the estimated parameters"""
    
    Params1_DMSO=Reference_parameters[:2]  #Reference parameter values for the first estimation step
    Params2_DMSO=Reference_parameters[2:]  #Reference parameter values for the second estimation step
    
    n1=np.sum(varying_parameters<2)  #number of varying parameters in the first step
    n2=np.sum(varying_parameters>=2) #number of varying parameters in the second step

    #now we build the list of parameters that won't change...
    c1=list(range(2))  #... in the first estimation step...
    c2=list(range(4))  #... and in the second one.
    B1=[]              #Bounds for the first estimation step
    B2=[]              #Bounds for the second estimation step
    for i in varying_parameters:
        if i in c1:
            c1.remove(i)
            B1+=[Bounds[i]]
        if i-2 in c2:
            c2.remove(i-2)
            B2+=[Bounds[i]]
    
    #Optimization parameters:
    nruns1=100 #number of independent runs of the first optimization step
    nruns2=500 #number of independent runs of the second optimization step
    LHS=False #should we perform LHS on initial guesses (apparently, it's not working very well with proportional error)
    maxiter=int(1e6)  #maximum duration (in evaluations of the model) of one run
    
    #Going to the details of the estimation:
    if n1>0:  #First estimation step
        d1=c1  #positions at which to insert the constant parameters
        if verbose_success or verbose_error:
            print(n1)
            print(c1)
            print(d1)
    
        opt1=CER.Sample_Estimate(PL.profile_likelihood,
                                 n1,
                                 (d1,Params1_DMSO[c1],EM.logLikelihood_ProportionalError,(Data[:,0],SCBe.S,t,Data[0,0,0])),
                                 nsamples=nruns1,
                                 lhs=LHS,
                                 maxeval=maxiter,
                                 bounds=B1,
                                 output_likelihood=True,
                                 verbose_success=verbose_success,
                                 verbose_error=verbose_error)
        params1=np.insert(opt1['parameters'],d1,Params1_DMSO[c1])
        l1=opt1['error']
    else:
        params1=Params1_DMSO
        l1=References_likelihoods[0]
    deltaSC_tmp=params1[0]+0.5*np.log(Data[0,0,0])  #the relationship between RhoS and DeltaSc is unaffected by the drug treatments.
    other_params=np.array([params1[0],deltaSC_tmp])

    if n2>0:  #Second estimation step
        if c2==[]:
            d2=[]
        else:
            d2=[c2[0]]
            for i in range(1,len(c2)):
                d2+=[d2[i-1]+c2[i]-c2[i-1]-1]
        if verbose_success or verbose_error:
            print(n2)
            print(c2)
            print(d2)
        
        opt2=CER.Sample_Estimate(PL.profile_likelihood,
                                 n2,
                                 (d2,Params2_DMSO[c2],EM.logLikelihood_ProportionalError,(Data[:,1:],SCBe.TB,t,[Data[0,0,0],0,0],other_params)),
                                 nsamples=nruns2,
                                 bounds=B2,
                                 lhs=LHS,
                                 maxeval=maxiter,
                                 output_likelihood=True,
                                 verbose_success=verbose_success,
                                 verbose_error=verbose_error)
        params2=np.insert(opt2['parameters'],d2,Params2_DMSO[c2])
        l2=opt2['error']
    else:
        params2=Params2_DMSO
        l2=References_likelihoods[1]
    
    params=np.concatenate((params1,params2))
    likelihood=l1+l2
    return({'parameters':params,'error':likelihood})
