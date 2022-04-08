
#!/usr/bin/python

import glam
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pandas as pd


class GLAM(object):
    """
    GLAM model instance that includes
    data, pymc3.model.Model instance,
    trace, parameter estimates,
    fit indices and predictions.
    """
    def __init__(self, data):
        super(GLAM, self).__init__()
        self.data = data
        self.n_items = len([col for col in data.columns
                            if col.startswith('item_value_')])

    def make_model_bf(self, kind, **kwargs):
        self.model = make_models(df=self.data, kind=kind, **kwargs)
        self.type = kind
        print('Generating models with free boundary parameter...  ')


    def fit(self, method='NUTS', **kwargs):
        self.trace = glam.fit.fit_models(self.model, method=method, **kwargs)
        self.estimates = glam.utils.extract_modes(self.trace)

    def compute_waic(self):
        if not isinstance(self.model, list):
            self.waic = pm.waic(trace=self.trace, model=self.model)
        else:
            self.waic = np.array([pm.waic(trace=trace, model=model)
                                 for (trace, model) in zip(self.trace, self.model)])

    def predict_bf(self, n_repeats=1, error_weight=0.05):

        prediction = pd.DataFrame()

        subjects = np.unique(self.data['subject'])

        for s, subject in enumerate(subjects):

            if self.type is 'individual':
                parameters = [self.estimates[s].get(parameter)
                              for parameter in ['v', 'gamma', 's', 'tau', 't0']]
                boundary = self.estimates[s].get('b')
            elif self.type is 'pooled':
                parameters = [self.estimates.get(parameter)
                              for parameter in ['v', 'gamma', 's', 'tau', 't0']]
                boundary = self.estimates.get('b')

            elif self.type is 'hierarchical':
                parameters = [self.estimates.get(parameter)[s]
                              for parameter in ['v', 'gamma', 's', 'tau', 't0']]
                boundary = self.estimates.get('b')[s]
            else:
                raise ValueError('Use .fit method first to obtain parameter estimates.')

            value_cols = ['item_value_{}'.format(i)
                          for i in range(self.n_items)]
            gaze_cols = ['gaze_{}'.format(i)
                         for i in range(self.n_items)]

            values = self.data[value_cols][self.data['subject'] == subject].values
            gaze = self.data[gaze_cols][self.data['subject'] == subject].values

            rt_min = self.data['rt'][self.data['subject'] == subject].values.min()
            rt_max = self.data['rt'][self.data['subject'] == subject].values.max()
            error_range = (rt_min, rt_max)

            subject_prediction = glam.simulation.simulate_subject(parameters,
                                                                  values,
                                                                  gaze,
                                                                  n_repeats=n_repeats,
                                                                  subject=subject,
                                                                  boundary=boundary,
                                                                  error_weight=error_weight,
                                                                  error_range=error_range)
            prediction = pd.concat([prediction, subject_prediction])

        self.prediction = prediction

    def exchange_data(self, new_data, verbose=True):
        if verbose:
            print('Replaced attached data ({} trials) with new data ({} trials)...'.format(len(self.data), len(new_data)))
        self.data = new_data


def make_models(df, kind, verbose=True, **kwargs):

    if kind == 'individual':
        data = glam.utils.format_data(df)
        if verbose:
            print('Generating single subject models for {} subjects...'.format(data['n_subjects']))
        models = []
        for s, subject in enumerate(data['subjects']):
            subject_model = make_subject_model(rts=data['rts'][data['subject_idx'] == subject],
                                               gaze=data['gaze'][data['subject_idx'] == subject],
                                               values=data['values'][data['subject_idx'] == subject],
                                               error_ll=data['error_lls'][s],
                                               **kwargs)
            models.append(subject_model)
        return models

    elif kind == 'pooled':
        if verbose:
            print('Generating pooled model including {} trials...'.format(len(df)))
        pooled = df.copy()
        pooled['subject'] = 0
        data = glam.utils.format_data(pooled)
        pooled_model = make_subject_model(rts=data['rts'],
                                          gaze=data['gaze'],
                                          values=data['values'],
                                          error_ll=data['error_lls'][0],
                                          **kwargs)
        return pooled_model

    elif kind == 'hierarchical':
        data = glam.utils.format_data(df)
        if verbose:
            print('Generating hierarchical model for {} subjects...'.format(data['n_subjects']))
        hierarchical_model = make_hierarchical_model(rts=data['rts'],
                                                     gaze=data['gaze'],
                                                     values=data['values'],
                                                     error_lls=data['error_lls'],
                                                     subject_idx=data['subject_idx'],
                                                     **kwargs)
        return hierarchical_model

    else:
        raise ValueError("'{}' model not Recognized. Use 'individual', 'pooled' or 'hierarchical'.".format(kind))


def make_subject_model(rts, gaze, values, error_ll,
                       b_val=None, # add fixed value for the b
                       v_val=None,
                       gamma_val=None,
                       s_val=None,
                       tau_val=None,
                       t0_val=None,
                       zerotol=1e-6, error_weight=0.05, boundary=1.,
                       gamma_bounds=(-1, 1)):
    with pm.Model() as glam_individual:

        # Mechanics
        #b = pm.Deterministic('b', tt.constant(boundary, dtype='float32'))
        p_error = pm.Deterministic('p_error', tt.constant(error_weight, dtype='float32'))

        # Parameter priors
        # Boundary as free parameter (PS modification 1/2021)
        if b_val is None:
            b = pm.Uniform('b', 0.5, 3, testval=1)
        else:
            b = pm.Deterministic('b', tt.ones(1)*b_val)
        ############################ 
        
        if v_val is None:
            v = pm.Uniform('v', zerotol, 0.01, testval=0.0002)
        else:
            v = pm.Deterministic('v', tt.ones(1)*v_val)

        if gamma_val is None:
            gamma = pm.Uniform('gamma', gamma_bounds[0], gamma_bounds[1], testval=0.3)
        else:
            gamma = pm.Deterministic('gamma', tt.ones(1)*gamma_val)

        if s_val is None:
            SNR = pm.Uniform('SNR', 1, 500, testval=80)
            s = pm.Deterministic('s', v*SNR)
        else:
            s = pm.Deterministic('s', tt.ones(1)*s_val)

        if tau_val is None:
            tau = pm.Uniform('tau', 0, 5, testval=1)
        else:
            tau = pm.Deterministic('tau', tt.ones(1)*tau_val)

        if t0_val is None:
            t0 = pm.Uniform('t0', 0, 500, testval=50)
        else:
            t0 = pm.Deterministic('t0', tt.ones(1)*t0_val)

        # Likelihood
        def lda_logp(rt,
                     gaze, values,
                     error_ll,
                     zerotol):

            # compute drifts
            drift = glam.components.expdrift(v, tau, gamma, values, gaze, zerotol)
            glam_ll = glam.components.tt_wienerrace_pdf(rt[:, None], drift, s, b, t0, zerotol)

            # mix likelihoods
            mixed_ll = ((1-p_error) * glam_ll +
                        p_error * error_ll)

            mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
            return tt.sum(tt.log(mixed_ll + zerotol))

        obs = pm.DensityDist('obs', logp=lda_logp,
                             observed=dict(rt=rts,
                                           gaze=gaze,
                                           values=values,
                                           error_ll=error_ll,
                                           zerotol=zerotol))
    return glam_individual


def make_hierarchical_model(rts, gaze, values, error_lls, subject_idx,
                            b_val=None, # add fixed value for the b
                            v_val=None,
                            gamma_val=None,
                            s_val=None,
                            tau_val=None,
                            t0_val=None,
                            zerotol=1e-6, error_weight=0.05, boundary=1.,
                            gamma_bounds=(-1, 1)):

    n_subjects = len(np.unique(subject_idx))
    subject_idx = subject_idx.astype(int)

    with pm.Model() as glam_hierarchical:

        # Mechanics
        #b = pm.Deterministic('b', tt.constant(boundary, dtype='float32'))
        p_error = pm.Deterministic('p_error', tt.constant(error_weight, dtype='float32'))

        # Parameter priors
        # Boundary as free parameter
        if b_val is None:
            b_mu = pm.Uniform('b_mu', 0.5, 3, testval=1)
            b_sd = pm.Uniform('b_sd', 0, 3, testval=1)
            b_bound = pm.Bound(pm.Normal, 0.5, 3)
            b = b_bound('b', mu=b_mu, sd=b_sd, shape=n_subjects)
        else:
            b = pm.Deterministic('b', tt.ones(n_subjects) * b_val)        
        ####################################
        
        if v_val is None:
            v_mu = pm.Uniform('v_mu', zerotol, 0.01, testval=0.0001)
            v_sd = pm.Uniform('v_sd', 0.00001, 0.01, testval=0.001)
            v_bound = pm.Bound(pm.Normal, 0, 0.01)
            v = v_bound('v', mu=v_mu, sd=v_sd, shape=n_subjects)
        else:
            v = pm.Deterministic('v', tt.ones(n_subjects) * v_val)

        if gamma_val is None:
            gamma_mu = pm.Uniform('gamma_mu', gamma_bounds[0], gamma_bounds[1], testval=0.5)
            gamma_sd = pm.Uniform('gamma_sd', zerotol, 2, testval=0.5)
            gamma_bound = pm.Bound(pm.Normal, gamma_bounds[0], gamma_bounds[1])
            gamma = gamma_bound('gamma', mu=gamma_mu, sd=gamma_sd, shape=n_subjects)
        else:
            gamma = pm.Deterministic('gamma', tt.ones(n_subjects) * gamma_val)

        if s_val is None:
            SNR_mu = pm.Uniform('SNR_mu', 1, 500, testval=150)
            SNR_sd = pm.Uniform('SNR_sd', 1, 500, testval=50)
            SNR_bound = pm.Bound(pm.Normal, 1, 500)
            SNR = SNR_bound('SNR', mu=SNR_mu, sd=SNR_sd, shape=n_subjects)
            s = pm.Deterministic('s', v*SNR)
        else:
            s = pm.Deterministic('s', tt.ones(n_subjects) * s_val)

        if tau_val is None:
            tau_mu = pm.Uniform('tau_mu', 0, 5, testval=1)
            tau_sd = pm.Uniform('tau_sd', zerotol, 5, testval=1)
            tau_bound = pm.Bound(pm.Normal, 0, 5)
            tau = tau_bound('tau', mu=tau_mu, sd=tau_sd, shape=n_subjects)
        else:
            tau = pm.Deterministic('tau', tt.ones(n_subjects) * tau_val)

        if t0_val is None:
            t0 = pm.Uniform('t0', 0, 500, testval=50, shape=n_subjects)
        else:
            t0 = pm.Deterministic('t0', tt.ones(n_subjects) * t0_val)

        # Likelihood
        def lda_logp(rt,
                     gaze, values,
                     error_lls,
                     zerotol):

            # compute drifts
            drift = glam.components.expdrift(v[subject_idx, None],
                                             tau[subject_idx, None],
                                             gamma[subject_idx, None],
                                             values, gaze, zerotol)
            glam_ll = glam.components.tt_wienerrace_pdf(rt[:, None],
                                                        drift,
                                                        s[subject_idx, None],
                                                        b[subject_idx, None],
                                                        t0[subject_idx, None],
                                                        zerotol)

            # mix likelihoods
            mixed_ll = ((1-p_error) * glam_ll +
                        p_error * error_lls[subject_idx])

            mixed_ll = tt.where(tt.isnan(mixed_ll), 0., mixed_ll)
            return tt.sum(tt.log(mixed_ll + zerotol))

        obs = pm.DensityDist('obs', logp=lda_logp,
                             observed=dict(rt=rts,
                                           gaze=gaze,
                                           values=values,
                                           error_lls=error_lls,
                                           zerotol=zerotol))
    return glam_hierarchical
