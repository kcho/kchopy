#!/data/pnl/kcho/anaconda3/bin/python

from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri


def pcor(var1, var2, covariate, method='spearman'):
    '''Run R ppcor's partial correlation

    Key arguments:
        var1, var2, covariate: float or int numpy array
        method: str, 'spearman' or 'pearson'

    '''
    # import ppcor library in R
    base = importr('ppcor')

    # define variables in R
    x = FloatVector(var1)
    y = FloatVector(var2)
    c = FloatVector(covariate)

    # assign values
    r.assign('x', x)
    r.assign('y', y)
    r.assign('c', c)

    # run partial correlation in R and return outputs to python
    r(f'pcorOut <- pcor.test(x, y, c, method = "{method}")')
    pcor_out = r('pcorOut')
    pcor_out_df = pandas2ri.rpy2py(pcor_out)

    return pcor_out_df


def example():
    '''Example run'''

    print('*Example run of partial_correlation.py\n')
    import numpy as np

    n1 = np.random.random(10)
    n2 = np.random.random(10)
    n3 = np.random.random(10)

    print('\tRandom float numpy arrays')
    print('\t', n1, n2, n3)
    pcor_out = pcor(n1, n2, n3)
    print(pcor_out)

    n1 = np.random.randint(0, 10, 10)
    n2 = np.random.randint(0, 10, 10)
    n3 = np.random.randint(0, 10, 10)

    print('\tRandom int numpy arrays')
    print('\t', n1, n2, n3)
    pcor_out = pcor(n1, n2, n3)
    print(pcor_out)


if __name__ == '__main__':
    example()
