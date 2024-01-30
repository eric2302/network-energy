import jpype as jp # pipe from java to python
import numpy as np

class Calculators():
    """
    Information-theoretic calculators class.

    Example usage (for mutual information):
        cal = Calculators()
        cal.activate()
        cal.mi_init('Kraskov')
        r = cal.mi_calc(s, t) # just the result
        r, pvalue, surrogate_dist = cal.mi_calc(s, t, surrogates=100) # result, p-value, and surrogate distribution
    """

    miCalc = None
    teCalc = None

    def __init__(self, jidt_path='infodynamics.jar'):
        self.activate(jidt_path)

    def activate(self, jidt_path='infodynamics.jar'):
        """
        Activate Java Virtual Environment and pipe to JIDT.
        """
        if not jp.isJVMStarted():
            jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=' + jidt_path)

    def mi_init(self, type_of_calculator='Gaussian'):
        match type_of_calculator:
            case 'Gaussian':
                self.miCalc = jp.JClass('infodynamics.measures.continuous.gaussian.MutualInfoCalculatorMultiVariateGaussian')()
            case 'Kraskov':
                self.miCalc = jp.JClass('infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1')()
            case _:
                raise('Type of calculator not recognised')
        

    def mi_calc(self, source, target, delay=0, surrogates=0):
        # TODO: not initilised error message
        self.miCalc.setProperty("TIME_DIFF", str(delay))
        self.miCalc.initialise()
        self.miCalc.setObservations(source, target)

        if surrogates > 0:
            null_dist = self.miCalc.computeSignificance(surrogates)
            surrogate_mi = np.array([x for x in null_dist.distribution]) # because of jpype's handling of arrays, we need to do the following
            pvalue = np.mean(null_dist.actualValue < surrogate_mi)
            return null_dist.actualValue, pvalue, surrogate_mi
        else:
            result = self.miCalc.computeAverageLocalOfObservations()
            return result
        
    def te_init(self, type_of_calculator='Kraskov'):
        match type_of_calculator:
            case 'Gaussian':
                self.teCalc = jp.JClass('infodynamics.measures.continuous.gaussian.TransferEntropyCalculatorGaussian')()
            case 'Kraskov':
                self.teCalc = jp.JClass('infodynamics.measures.continuous.kraskov.TransferEntropyCalculatorKraskov')()
            case _:
                raise('Type of calculator not recognised')

    def te_calc(self, source, target, delay=1, source_history_embedding=10, surrogates=0):
        # set properties to non-default values
        self.teCalc.setProperty("k_HISTORY", "1")
        self.teCalc.setProperty("k_TAU", "1")
        self.teCalc.setProperty("l_HISTORY", str(source_history_embedding))
        self.teCalc.setProperty("l_TAU", "1")
        self.teCalc.setProperty("DELAY", str(delay))
        # initialise and compute
        self.teCalc.initialise()
        self.teCalc.setObservations(source, target)

        if surrogates > 0:
            null_dist = self.teCalc.computeSignificance(surrogates)
            surrogate_te = np.array([x for x in null_dist.distribution]) # because of jpype's handling of arrays, we need to do the following
            pvalue = np.mean(null_dist.actualValue < surrogate_te)
            return null_dist.actualValue, pvalue, surrogate_te

        else:
            result = self.teCalc.computeAverageLocalOfObservations()
            return result
