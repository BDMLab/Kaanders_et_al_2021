import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import errno

def compute_gaze_influence(data):
    """
    """
    import statsmodels.api as sm

    # calculate relative item value of left over mean of other options
    if 'value_left_minus_mean_others' not in data.columns:
        data['value_left_minus_mean_others'] = data['item_value_0'] - data['item_value_1']
    
    # -- Not used in case of two items (PSD, 11/2018)
    ## calculate value range of other options
    #data['value_range_others'] = np.abs(data['item_value_1'] - data['item_value_2'])
    
    
    # Add indicator column for left choices
    data['left_chosen'] = data['choice'] == 0
    # Calculate relative gaze advantage of left over other options
    data['gaze_left_minus_mean_others'] = data['gaze_0'] - data['gaze_1'] 
    # Add indicator column for trials with longer gaze towards left option
    data['left_longer'] = data['gaze_left_minus_mean_others'] > 0

    data_out = pd.DataFrame()

    for s, subject in enumerate(data['subject'].unique()):
        subject_data = data[data['subject'] == subject].copy()

        X = subject_data[['value_left_minus_mean_others']]
        X = sm.add_constant(X)
        y = subject_data['left_chosen']

        logit = sm.Logit(y, X)

        result = logit.fit(disp=0)
        predicted_pchooseleft = result.predict(X)

        subject_data['corrected_choice'] = subject_data['left_chosen'] - predicted_pchooseleft
        data_out = pd.concat([data_out, subject_data])

    # Compute difference in corrected P(choose left) between positive and negative final gaze advantage
    tmp = data_out.groupby(['subject', 'left_longer']).corrected_choice.mean().unstack()
    gaze_influence = (tmp[True] - tmp[False]).values

    return gaze_influence


def compute_mean_rt(df):
    """
    Computes subject wise mean RT
    """
    return df.groupby('subject').rt.mean().values


def compute_p_choose_best(df):
    """
    Computes subject wise P(choose best)
    """
    if 'best_chosen' not in df.columns:
        values = df[[c for c in df.columns if c.startswith('item_value')]].values
        choices = df['choice'].values
        best_chosen = (values.argmax(axis=1) == choices).astype('int')
        df['best_chosen'] = best_chosen
    return df.groupby('subject').best_chosen.mean().values


# Modification for dislike case (PS mod. May 2019)
def compute_p_choose_worst(df):
    """
    Computes subject wise P(choose best)
    """
    if 'best_chosen' not in df.columns:
        values = df[[c for c in df.columns if c.startswith('item_value')]].values
        choices = df['choice'].values
        # Best chosen in this case is the opposite to the best option, given that we have only two alternatives 
        best_chosen = (values.argmax(axis=1) != choices).astype('int')
        df['best_chosen'] = best_chosen
    return df.groupby('subject').best_chosen.mean().values



def run_linear_model(x, y, verbose=True):

    X = sm.add_constant(x)
    lm = sm.OLS(y, X).fit()

    if verbose:
        print(lm.summary())
        print('Slope = {:.2f}'.format(lm.params[-1]))
        print('t({:d}) = {:.2f}'.format(int(lm.df_resid), lm.tvalues[-1]))
        print('P = {:.10f}'.format(lm.pvalues[-1]))
    return lm


def q1(series):
    q1 = series.quantile(0.25)
    return q1


def q3(series):
    q3 = series.quantile(0.75)
    return q3


def iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return IQR


def std(series):
    sd = series.std(ddof=0)
    return sd


def se(series):
    n = len(series)
    se = series.std() / np.sqrt(n)
    return se


def sci_notation(num, exact=False, decimal_digits=1, precision=None, exponent=None):
    """
    exact keyword toggles between 
        - exact reporting with coefficient
        - reporting next higher power of 10 for P < 10^XX reporting

    https://stackoverflow.com/a/18313780
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    from math import floor, log10
    if not exponent:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits
    
    if exact:
        return r"${0:.{2}f}\times10^{{{1:d}}}$".format(coeff, exponent, precision)
    else:
        return r"$10^{{{0}}}$".format(exponent+1)


def write_summary(lm, filename):
    """
    Write statsmodels lm summary as to file as csv.
    """
    predictors = lm.model.exog_names
    tvals = lm.tvalues
    pvals = lm.pvalues
    betas = lm.params
    se = lm.bse
    ci = lm.conf_int()
    r2 = lm.rsquared
    r2a = lm.rsquared_adj
    aic = lm.aic
    bic = lm.bic
    ll = lm.llf
    F = lm.fvalue
    df = lm.df_resid
    n = lm.nobs
    
    table = pd.DataFrame(dict(predictors=predictors,
                              tvals=tvals,
                              pvals=pvals,
                              betas=betas,
                              se=se,
                              ci0=ci[0],
                              ci1=ci[1],
                              df=df,
                              n=n,
                              r2=r2))
    table.to_csv(filename)
    

def make_sure_path_exists(path):
    """
    Used to check or create existing folder structure for results.
    https://stackoverflow.com/a/5032238
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise