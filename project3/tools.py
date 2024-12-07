import numpy as np, pandas as pd
from numpy import random
from numpy import linalg as la
from scipy import optimize
from scipy.stats import norm
from tabulate import tabulate

name = 'Logit'

DOCHECKS = True 

def G(z): 
    Gz = 1. / (1. + np.exp(-z))
    return Gz

def q(theta, y, x): 
    return -loglikelihood(theta, y, x)

def loglikelihood(theta, y, x):

    if DOCHECKS: 
        assert np.isin(y, [0,1]).all(), f'y must be binary: found non-binary elements.'
        assert y.ndim == 1
        assert x.ndim == 2 
        N,K = x.shape 
        assert y.size == N
        assert theta.ndim == 1 
        assert theta.size == K 

    # 0. unpack parameters 
    # (trivial, we are just estimating the coefficients on x)
    beta = theta 
    
    # 1. latent index
    z = x@beta
    Gxb = G(z)
    
    # 2. avoid log(0.0) errors
    h = 1e-8 # a tiny number 
    Gxb = np.fmax(Gxb, h)     # truncate below at 1e-8 
    Gxb = np.fmin(Gxb, 1.0-h) # truncate above at 0.99999999

    ll = (y==1)*np.log(Gxb) + (y==0)*np.log(1.0 - Gxb) 
    return ll

def Ginv(u): 
    '''Inverse logistic cdf: u should be in (0;1)'''
    x = - np.log( (1.0-u) / u )
    return x

def starting_values(y,x): 
    b_ols = la.solve(x.T@x, x.T@y)
    return b_ols*4.0

def predict(theta, x): 
    # the "prediction" is the response probability, Pr(y=1|x)
    yhat = G(x@theta) 
    return yhat 

def sim_data(theta: np.ndarray, N:int): 
    '''sim_data: simulate a dataset of size N with true K-parameter theta

    Args. 
        theta: (K,) vector of true parameters (k=0 will always be a constant)
        N (int): number of observations to simulate 
    
    Returns
        tuple: y,x
            y (float): binary outcome taking values 0.0 and 1.0
            x: (N,K) matrix of explanatory variables
    '''
    
    # 0. unpack parameters from theta
    # (trivial, only beta parameters)
    beta = theta

    K = theta.size 
    assert K>1, f'Only implemented for K >= 2'
    
    # 1. simulate x variables, adding a constant 
    oo = np.ones((N,1))
    xx = np.random.normal(size=(N,K-1))
    x  = np.hstack([oo, xx]);
    
    # 2. simulate y values
    
    # 2.a draw error terms 
    uniforms = np.random.uniform(size=(N,))
    u = Ginv(uniforms)

    # 2.b compute latent index 
    ystar = x@beta + u
    
    # 2.b compute observed y (as a float)
    y = (ystar>=0).astype(float)

    # 3. return 
    return y, x


def average_partial_effect(x_i, betas, cov_matrix, k=1):
    '''
    Compute the average partial effect of a binary variable in the logit model.
    '''
    # Get the observations where the binary variable is 0 and 1
    x_ij = x_i[np.where(x_i[:, k]==0),:].reshape(-1, x_i.shape[1])
    x_i_mj = x_ij.copy()
    x_i_mj[:, k] = 1  # Keep everythin the same, but change race to 1 for all obs. 

    # calculate the partial effect for each observation and the average partial effect
    gx0 = 1/(1+np.exp(-(x_ij @ betas))) 
    gx1 = 1/(1+np.exp(-(x_i_mj @ betas)))
    pe_i =  gx1 - gx0
    ape = np.mean(pe_i)
    sample_std = np.std(pe_i)
    
    # Compute the derivate of g(x) with respect to beta
    g_x0_prime = gx0*(1-gx0)
    g_x1_prime = gx1*(1-gx1)
    
    # Compute the gradient of the average partial effect
    grad_i = g_x0_prime[:, None] * x_i_mj - g_x1_prime[:, None] * x_ij
    grad = np.mean(grad_i, axis=0)
    
    # Compute the covariance matrix of the parameter estimates
    pe_cov_matrix = grad[:,None].T @ cov_matrix @ grad[:,None]
    
    return ape, pe_cov_matrix, sample_std

def dataframe_to_latex_table(df, caption='', label=''):
    """
    Converts a pandas DataFrame with MultiIndex (sRace, Model, Tipo) and regressors as columns
    to a LaTeX table that exactly mirrors the DataFrame's structure.
    
    Rows with standard deviations (Tipo = 'Std') are shown in parentheses directly below coefficients.

    Parameters:
    - df: pandas DataFrame with MultiIndex (sRace, Model, Tipo).
    - caption: str. Caption for the LaTeX table.
    - label: str. Label for the LaTeX table.

    Returns:
    - str: LaTeX table as a string.
    """
    # Start LaTeX table
    col_format = 'l l' + ' c' * len(df.columns)  # First column for index, others for regressors
    latex = '\\begin{table}[H]\n'
    latex += '    \\centering\n'
    latex += f'    \\begin{{tabular}}{{{col_format}}}\n'
    latex += '\\toprule\n'

    # Header row: Regressors
    header = ' & ' + ' & '.join([f'\\textbf{{{col}}}' for col in df.columns]) + ' \\\\\n'
    latex += header
    latex += '\\midrule\n'

    # Iterate over `sRace` and `Model` in MultiIndex
    current_sRace = None
    for (sRace, model, tipo), row in df.iterrows():
        if current_sRace != sRace:
            if current_sRace is not None:
                latex += '\\midrule\n'  # Add a horizontal line between groups
            latex += f'{sRace} '
            current_sRace = sRace

        # Format the `Model` and row values
        if tipo == 'Coeff':
            latex += f'& {model:<15} & ' + ' & '.join([f"${v:.3f}$" if pd.notna(v) else '' for v in row]) + ' \\\\\n'
        elif tipo == 'Std':
            latex += f'& {"":<15} & ' + ' & '.join([f"(${v:.3f})$" if pd.notna(v) else '' for v in row]) + ' \\\\\n'

    # End LaTeX table
    latex += '\\bottomrule\n'
    latex += '    \\end{tabular}\n'
    latex += f'    \\caption{{{caption}}}\n'
    latex += f'    \\label{{{label}}}\n'
    latex += '\\end{table}\n'

    return latex

def dataframe_to_latex_table_multirow(df, caption='', label=''):
    """
    Converts a pandas DataFrame with MultiIndex (sRace, Model, Tipo) and regressors as columns
    to a LaTeX table where coefficients and their standard deviations share a row using \multirow.

    Parameters:
    - df: pandas DataFrame with MultiIndex (sRace, Model, Tipo).
    - caption: str. Caption for the LaTeX table.
    - label: str. Label for the LaTeX table.

    Returns:
    - str: LaTeX table as a string.
    """
    # Start LaTeX table
    col_format = 'l l' + ' c' * len(df.columns)  # First two columns for indices, others for regressors
    latex = '\\begin{table}[H]\n'
    latex += '    \\centering\n'
    latex += f'    \\begin{{tabular}}{{{col_format}}}\n'
    latex += '\\toprule\n'

    # Header row: Regressors
    header = ' & ' + ' & '.join([f'\\textbf{{{col}}}' for col in df.columns]) + ' \\\\\n'
    latex += header
    latex += '\\midrule\n'

    # Iterate over `sRace` and `Model` in MultiIndex
    current_sRace = None
    for sRace, group in df.groupby(level='sRace'):
        if current_sRace != sRace:
            if current_sRace is not None:
                latex += '\\midrule\n'  # Add a horizontal line between groups
            current_sRace = sRace

        for i, (model, model_group) in enumerate(group.groupby(level='Model')):
            latex_race = ' ' if i > 0 else f'\\textbf{{{current_sRace}}} '  # Add a header for the new group
            # Extract coefficient and standard deviation rows for the model
            coeff_row = model_group.xs('Coeff', level='Tipo')
            std_row = model_group.xs('Std', level='Tipo')

            # Start the row with \multirow for the Model
            model = '\\vspace{3pt} ' + model
            latex += f'\\multirow{{2}}{{*}} {{{model}}} &'
            latex += ' & '.join([f"${v:.3f}$" if pd.notna(v) else '' for v in coeff_row.values[0]]) + '\\vspace{-3pt} \\\\\n'

            # Add the second row for the standard deviation
            latex += f' &'
            latex += ' & '.join([f"(${v:.3f})$" if pd.notna(v) else '' for v in std_row.values[0]]) + ' \\\\\n'

    # End LaTeX table
    latex += '\\bottomrule\n'
    latex += '    \\end{tabular}\n'
    latex += f'    \\caption{{{caption}}}\n'
    latex += f'    \\label{{{label}}}\n'
    latex += '\\end{table}\n'

    return latex
