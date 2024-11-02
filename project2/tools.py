import pandas as pd
import numpy as np
from numpy import linalg as la
from tabulate import tabulate
from scipy.stats import chi2, norm
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso

class penalty_term:
    def __init__(self, X:np.ndarray, y:np.ndarray, alpha:float, c:float, n:int, p:int):
        self.n = n
        self.p = p
        self.X = X
        self.y = y
        self.alpha = alpha
        self.c = c
    
    def _max_term(self, func:callable, **kwargs):
        # calculate the max term
        max_term = {}
        for j in range(self.X.shape[1]):
            max_term[j] = func(self.X[:,j], **kwargs)
        return max(max_term.values())
    
    def _scale_factor(self, ):
        return 2*self.c/np.sqrt(self.n)
    
    def _quantile_factor(self, ):
        return norm.ppf(1-self.alpha/(2*self.p))

    def brt_rule(self, sigma:float):
        scale = self._scale_factor() * sigma
        quantile = self._quantile_factor()
        max_term = self._max_term(lambda x: np.sqrt(np.mean(x**2)))
        return scale*quantile*max_term
    def bcch_pilot_rule(self):
        scale = self._scale_factor()
        quantile = self._quantile_factor()
        max_term = self._max_term(func=(lambda x, y: np.sqrt(np.mean((y-y.mean())**2*x**2))), y=self.y)
        return scale*quantile*max_term
    def bcch_rule(self, residuals:np.ndarray):
        scale = self._scale_factor() # scale factor
        quantile = self._quantile_factor()
        max_term = self._max_term(func=(lambda x, y: np.sqrt(np.mean((y-y.mean())**2*x**2))), y=residuals)
        return scale*quantile*max_term

class MyLasso_123:
    def __init__(self, X, y, xlabels:list=None, max_iter:int=10_000, tol:float=1e-4, fit_intercept:bool=False):
        self.X = X
        self.xlabels = xlabels
        self.y = y
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
    
    def lasso(self, lambda_):
        """ """
        # Lasso regression
        lasso = Lasso(alpha=lambda_, max_iter=self.max_iter,fit_intercept=self.fit_intercept, tol=self.tol)
        fit = lasso.fit(self.X, self.y)
        fit.feature_names_in_ = self.xlabels
        return fit
    
def plot_lasso_path(penalty_grid, coefs, legends, vlines: dict = None):
    """
    Plots the coefficients as a function of the penalty parameter for Lasso regression.

    Parameters:
    penalty_grid (array-like): The penalty parameter values.
    coefs (array-like): The estimated coefficients for each penalty value.
    legends (list): The labels for each coefficient estimate.
    vlines (dict, optional): A dictionary of vertical lines to add to the plot. The keys are the names of the lines and the values are the penalty values where the lines should be drawn.
    
    """
    # Initiate figure 
    fig, ax = plt.subplots()

    # Plot coefficients as a function of the penalty parameter
    ax.plot(penalty_grid, coefs)

    # Set log scale for the x-axis
    ax.set_xscale('log')

    # Add labels
    plt.xlabel('Penalty, $\lambda$')
    plt.ylabel(r'Estimates, $\widehat{\beta}_j(\lambda)$')
    plt.title('Lasso Path')

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legends
    # lgd=ax.legend(legends,loc=(1.04,0))

    # set x lim 
    ax.set_xlim([min(penalty_grid), max(penalty_grid)])
    
    # Add vertical lines
    if vlines is not None:
        for name, penalty in vlines.items():
            ax.axvline(x=penalty, linestyle='--', color='grey')
            plt.text(penalty,
                     ax.get_ylim()[1]*0.85,
                     name,rotation=90)

    # Display plot
    plt.show()
    plt.close()

def estimate( 
        y: np.ndarray, 
        x: np.ndarray, 
        transform='', 
        T:int=None,
        robust:bool=False
    ) -> list:
    """Uses the provided estimator (mostly OLS for now, and therefore we do 
    not need to provide the estimator) to perform a regression of y on x, 
    and provides all other necessary statistics such as standard errors, 
    t-values etc.  

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)
        >> transform (str, optional): Defaults to ''. If the data is 
        transformed in any way, the following transformations are allowed:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation.
        >>t (int, optional): If panel data, t is the number of time periods in
        the panel, and is used for estimating the variance. Defaults to None.

    Returns:
        list: Returns a dictionary with the following variables:
        'b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov'
    """
    
    b_hat = est_ols(y, x)  # Estimated coefficients
    u_hat = y - x@b_hat  # Calculated residuals
    SSR = u_hat.T@u_hat  # Sum of squared residuals
    SST = (y - np.mean(y)).T@(y - np.mean(y))  # Total sum of squares
    R2 = 1 - SSR/SST
    
    sigma2, cov, se = variance(transform, SSR, x, T, u_hat, robust)
    t_values = b_hat/se
    
    names = ['b_hat', 'se', 'sigma2', 't_values', 'R2', 'cov', 'u_hat', 'rob']
    results = [b_hat, se, sigma2, t_values, R2, cov, u_hat, robust]
    return dict(zip(names, results))

def est_ols( y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimates y on x by ordinary least squares, returns coefficents

    Args:
        >> y (np.ndarray): Dependent variable (Needs to have shape 2D shape)
        >> x (np.ndarray): Independent variable (Needs to have shape 2D shape)

    Returns:
        np.array: Estimated beta coefficients.
    """
    return la.inv(x.T@x)@(x.T@y)

def variance(
        transform: str, 
        SSR: float, 
        x: np.ndarray, 
        T: int, 
        u_hat: np.ndarray,  # Adding residuals for robust SE calculation
        robust: bool = False  # Optional argument to switch to robust variance calculation
    ) -> tuple:
    """Calculates the covariance and standard errors from the OLS
    estimation, supporting robust (heteroskedasticity-consistent) variance.

    Args:
        >> transform (str): Specifies the data transformation:
            '': No transformations
            'fd': First-difference
            'be': Between transformation
            'fe': Within transformation
            're': Random effects estimation
        >> SSR (float): Sum of squared residuals.
        >> x (np.ndarray): Independent variables from regression.
        >> u_hat (np.ndarray): Residuals from regression for robust variance.
        >> T (int): The number of time periods.
        >> robust (bool): If True, calculates robust standard errors.

    Raises:
        Exception: If an invalid transformation is provided, an error is returned.

    Returns:
        tuple: Returns the error variance (mean square error),
               covariance matrix, and standard errors.
    """
    
    # Store n and k, used for DF adjustments.
    K = x.shape[1]
    if transform in ('', 'fd', 'be'):
        N = x.shape[0]
    else:
        N = x.shape[0] / T  # For panel data
    
    # Calculate sigma2 (error variance) using different methods based on the transformation
    if transform in ('', 'fd', 'be'):
        sigma2 = (np.array(SSR / (N - K)))
    elif transform.lower() == 'fe':
        sigma2 = np.array(SSR / (N * (T - 1) - K))
    elif transform.lower() == 're':
        sigma2 = np.array(SSR / (T * N - K))
    else:
        raise Exception('Invalid transform provided.')

    # Variance-covariance matrix calculation
    if robust: 
        # Meat and bread 
        u_hat_squared = np.diag(u_hat.flatten() ** 2) # flatten to 1-D array, so we create a diag matrix. 
        meat = x.T @ u_hat_squared @ x

        bread = la.inv(x.T @ x)

        cov = bread @ meat @ bread # Sandwich formula
    else:  
        cov = sigma2 * la.inv(x.T @ x) # Standard formula

    # Standard errors are the square root of the diagonal of the variance-covariance matrix
    se = np.sqrt(cov.diagonal()).reshape(-1, 1)
    
    return sigma2, cov, se

def wald_test(b_hat: np.ndarray, cov_mat: np.ndarray, R: np.ndarray, r: np.ndarray, verbose: bool = True, alpha:float=0.05) -> tuple:
    """
    Perform Wald test for the null hypothesis R * beta = r.

    Args:
        beta_hat (np.ndarray): Estimated coefficients (beta).
        avar_beta (np.ndarray): Covariance matrix of beta.
        R (np.ndarray): Matrix for linear hypothesis (e.g., [1, 1]).
        r (np.ndarray): Vector for linear hypothesis (e.g., [0]).
        verbose (bool): If True, print the results of the test.
        alpha (float): Significance level for the test.

    Returns:
        W_stat (float): The Wald test statistic.
        p_value (float): The p-value for the test.
    """
    # Compute R * beta_hat - r
    R_beta_diff = R @ b_hat - r
    
    # Compute Wald statistic: W = (Rβ - r)' [R Avar(β) R']⁻¹ (Rβ - r)
    W_stat = (R_beta_diff.T @ la.inv(R @ cov_mat @ R.T) @ R_beta_diff).item()
    
    # Degrees of freedom (number of restrictions)
    Q = R.shape[0]
    
    # Compute p-value from chi-squared distribution
    p_value = 1 - chi2.cdf(W_stat, df=Q)

    if verbose:
        if p_value < alpha:
            print(f"On a 5% significance level, we reject the null hypothesis.")
        else:
            print(f"On a 5% significance level, we cannot reject the null hypothesis.")
    
    
    return W_stat, p_value

def transform_output(
        labels: tuple,
        results: dict,
        headers=["coef", "std err", "t-values", "p-values", "conf-int1", "conf-int2", "significant"],
        alpha:float=0.05,
        **kwargs
):
    """ Transform the output of the OLS regression into a pandas dataframe
            Args:
                labels (tuple): Tuple with first a label for y, and then a list of labels for x.
                results (dict): The results from a regression. Needs to be in a dictionary with at least the following keys:
                    'b_hat', 'se', 't_values', 'R2', 'sigma2'   
                headers (list, optional): Column headers. Defaults to 
                    ["coef", "std err", "t-values", "p-values", "conf-int1", "conf-int2", "significant"]
                alpha (float, optional): Significance level for the p-values. Defaults to 0.05.
            Returns:
                pd.DataFrame: Returns a pandas dataframe with the results of the regression.
    """
    # unpack the labels
    label_y, label_x = labels
    coefs = results['b_hat'].flatten()
    std_err = results['se'].flatten()
    t_values = results['t_values'].flatten()
    
    # calculate p-values
    p_values = (1 - norm.cdf(np.abs(t_values))) * 2
    # calculate confidence intervals
    conf_int_low = coefs - abs(norm.ppf(alpha/2)) * std_err
    conf_int_up = coefs + abs(norm.ppf(alpha/2)) * std_err

    # create the output dataframe
    output = (pd.DataFrame(index=label_x,
                            data=({
                                'coef':coefs,
                                'std err': std_err,
                                't-values': t_values,
                                'p-values': p_values,
                                'conf-int1': conf_int_low,
                                'conf-int2': conf_int_up,
                                'significant': ['*' if val < alpha else '' for val in p_values]
                                })))
    # select the columns based on the headers
    output = output.loc[:,headers]

    return output

def reg_to_latex(
        filename, 
        output:pd.DataFrame,
        r2:float, 
        caption:str='',
        label:str='tab: x1y2z3'
        ):
    """ 
    Write multiple regression outputs to a LaTeX table
    
        Args:
            outputs (list): list of outputs from the regression
            names (list): list of names for the outputs
        Returns:
            None
        """
    if filename[-4:] != '.tex':
        filename += '.tex'
    
    headers = ['coef', 'std err', 't-values', 'p-values']
    columns = 'c | ' + 'c' * len(headers)
    
    with open(filename, 'w') as f:
        f.write('\\begin{table}[H]\n')
        f.write('\\centering\n')

        f.write(f'\\begin{{tabular}}{{{columns}}}\n')
        f.write(' & ' + ' & '.join([f'\\textbf{{{head}}}' for head in headers]) + ' \\\\\n')  
        f.write('\\toprule \\\n')

        for idx, row in output.iterrows():
            coefs = f"${row['coef']:.2f}$"
            std_devs = f"${row['std err']:.2f}$"
            t_values = f"${row['t-values']:.2f}$"
            p_values = f"${row['p-values']:.2f}$"
            f.write(f'{idx:10} &  {coefs} & {std_devs} & {t_values} & {p_values}  \\\\\n')

        f.write('\midrule \n')
        f.write(f'$R^2$ & {r2:.2f} {"&".join(["" for _ in range(len(headers)-1)])} \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\end{tabular}\n')
        f.write(f'\caption{{{caption}}}\n')
        f.write(f'\label{{{label}}}\n')
        f.write('\end{table}')


def print_table(
        labels: tuple,
        results: dict,
        headers=["", "Beta", "Se", "t-values", "p-values"],
        title="Results",
        _lambda:float=None,
        **kwargs
    ) -> None:
    """Prints a nice looking table, must at least have coefficients, 
    standard errors and t-values. The number of coefficients must be the
    same length as the labels.

    Args:
        >> labels (tuple): Touple with first a label for y, and then a list of 
        labels for x.
        >> results (dict): The results from a regression. Needs to be in a 
        dictionary with at least the following keys:
            'b_hat', 'se', 't_values', 'R2', 'sigma2'
        >> headers (list, optional): Column headers. Defaults to 
        ["", "Beta", "Se", "t-values"].
        >> title (str, optional): Table title. Defaults to "Results".
        _lambda (float, optional): Only used with Random effects. 
        Defaults to None.
    """
    
    # Unpack the labels
    label_y, label_x = labels
    
    # Create table, using the label for x to get a variable's coefficient,
    # standard error and t_value.
    table = []
    for i, name in enumerate(label_x):
        row = [
            name, 
            results.get('b_hat')[i], 
            results.get('se')[i], 
            results.get('t_values')[i], 
            (1-(norm.cdf(np.abs(results.get('t_values')[i]))))*2
        ]
        table.append(row)
    
    # Print the table
    print('---------------------------------------------')
    print(title)
    print(f"Dependent variable: {label_y}\n")
    print(tabulate(table, headers, **kwargs))
    
    # Print extra statistics of the model.
    print(f"R\u00b2 = {results.get('R2').item():.3f}")
    print(f"\u03C3\u00b2 = {results.get('sigma2').item():.3f}")
    if _lambda: 
        print(f'\u03bb = {_lambda.item():.3f}')
    print(f'Robust standard errors: {results.get("rob")}')

def breusch_pagan_test(u_hat: np.ndarray, x: np.ndarray) -> tuple:
    """
    Perform the Breusch-Pagan test for homoskedasticity, see W. p. 126. 

    Assumptions:
        Constant conditional fourth moment of u_hat (homokurtosis)
    
    Args:
        u_hat (np.ndarray): Residuals from the regression.
        x (np.ndarray): Independent variables with intercept as first column. 
    
    Returns:
        tuple: Returns the Breusch-Pagan test statistic and p-value.
    """

    assert np.any(np.all(x == 1, axis=0)), "The matrix x must include a constant (intercept) column."

    # 1. create the auxiliary regression by regressing squared residuals on x
    u_hat_squared = u_hat**2
    N, K = x.shape
    
    results = estimate(u_hat_squared, 
                       x, 
                       robust = False) # assumption
    
    # 2. calculate Breusch-Pagan statistic
    R2 = results['R2']
    bp_stat = R2 * N 

    # 3. calculate the p-value with chi-square distribution with df = K - 1
    p_value = 1 - chi2.cdf(bp_stat, df= K - 1)
    
    return bp_stat, p_value


# def remove_zero_columns(x, label_x):
#     """
#     The function removes columns from a matrix that are all zeros and returns the updated matrix and
#     corresponding labels.
    
#     Args:
#       x: The parameter `x` is a numpy array representing a matrix with columns that may contain zeros.
#       label_x: The parameter `label_x` is a list that contains the labels for each column in the input
#     array `x`.
    
#     Returns:
#       x_nonzero: numpy array of x with columns that are all zeros removed.
#       label_nonzero: list of labels for each column in x_nonzero.
#     """
    
#     # Find the columns that are not all close to zeros
#     nonzero_cols = ~np.all(np.isclose(x,0), axis=0)
    
#     # Remove the columns that are all zeros
#     x_nonzero = x[:, nonzero_cols]
    
#     # Get the labels for the columns that are not all zeros
#     label_nonzero = [label_x[i] for i in range(len(label_x)) if nonzero_cols[i]]
#     return x_nonzero, label_nonzero

# def demeaning_matrix(T:int):
#     """ create transformation matrix for within transformation to demean the data.
#     Args:   
#         T (int): Number of time periods    
#     Returns:
#         np.ndarray: T x T matrix
#     """
#     Q_T =  np.eye(T) - np.ones((T,T))/T
#     return Q_T

# def fd_matrix(T:int):
#     """ create transformation matrix for first-difference transformation of the data.
#     Args:
#         T (int): Number of time periods
#     Returns:
#         np.ndarray: (T-1)xT matrix
#     """
#     # Initialize a (T-1) x T matrix filled with zeros
#     D_T = np.zeros((T-1, T))
    
#     # Fill the matrix according to the first-difference structure
#     for i in range(T-1):
#         D_T[i, i] = -1
#         D_T[i, i+1] = 1
    
#     return D_T #(T-1)xT

# def perm(Q_T: np.ndarray, A: np.ndarray) -> np.ndarray:
#     """Takes a transformation matrix and performs the transformation on 
#     the given vector or matrix.

#     Args:
#         Q_T (np.ndarray): The transformation matrix. Needs to have the same
#         dimensions as number of years a person is in the sample.
        
#         A (np.ndarray): The vector or matrix that is to be transformed. Has
#         to be a 2d array.

#     Returns:
#         np.array: Returns the transformed vector or matrix.
#     """
#     # We can infer t from the shape of the transformation matrix.
#     M,T = Q_T.shape 
#     N = int(A.shape[0]/T)
#     K = A.shape[1]

#     # initialize output 
#     Z = np.empty((M*N, K))
    
#     for i in range(N): 
#         ii_A = slice(i*T, (i+1)*T)
#         ii_Z = slice(i*M, (i+1)*M)
#         Z[ii_Z, :] = Q_T @ A[ii_A, :]

#     return Z