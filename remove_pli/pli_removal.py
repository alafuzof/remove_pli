import numpy as np
import scipy.signal
import numba


def remove_pli(data, sampling_rate, num_harmonics, B, P, W, ac_frequency=None,
               mode='clean', use_numba=True):
    data = np.asarray(data, dtype=np.double)
    B = np.asarray(B, dtype=np.double)
    P = np.asarray(P, dtype=np.double)
    W = np.double(W)

    nyq = sampling_rate // 2

    data = data - data.mean()

    # IIR Bandpass filtering:
    corner_freqs = np.array((40.0, 70.0))  # Default wide band
    if ac_frequency is not None:
        ac_frequency = np.atleast_1d(ac_frequency)
        if len(ac_frequency) == 2:
            corner_freqs = ac_frequency
        elif ac_frequency == 50:
            corner_freqs = np.array((48.0, 52.0))
        elif ac_frequency == 60:
            corner_freqs = np.array((58.0, 62.0))

    order = 2  # Order
    sos = scipy.signal.butter(order, corner_freqs/nyq,
                              btype='bandpass',
                              analog=False,
                              output='sos')
    filtered_diff = scipy.signal.sosfilt(sos, data).astype(np.double)
    filtered_diff[1:] = np.diff(filtered_diff)
    filtered_diff[0] = 0.0

    if use_numba:
        s = _inner_loop_numba(data, filtered_diff, sampling_rate, num_harmonics, B, P, W)
    else:
        s = _inner_loop(data, filtered_diff, sampling_rate, num_harmonics, B, P, W)

    if mode == 'clean':
        data -= s
        return data
    elif mode == 'pli':
        return s


def _inner_loop(data, x_f, sampling_rate, num_harmonics, B, P, W):
    N = len(data)
    pli_data = np.zeros_like(data, dtype=np.double)
    nyq = sampling_rate // 2

    # 3dB cutoff bandwidth
    alpha_f = ((1.0 - np.arctan(np.pi*B[0]/sampling_rate)) /
               (1.0 + np.arctan(np.pi*B[0]/sampling_rate)))     # initial, \alpha_0
    alpha_inf = ((1.0 - np.tan(np.pi*B[1]/sampling_rate)) /
                 (1.0 + np.tan(np.pi*B[1]/sampling_rate)))      # asymptotic
    alpha_st = np.exp(np.log(0.05)/(B[2]*sampling_rate+1))  # rate of change

    # frequency estimator's forgetting factors
    lambda_f, lambda_inf, lambda_st = np.exp(np.log(0.05)/(P*sampling_rate+1))
    # lambda_inf = math.exp(math.log(0.05)/P[1]*sampling_rate+1)
    # lambda_st  = math.exp(math.log(0.05)/P[2]*sampling_rate+1)

    # Smoothing parameter (cut-off freq set at 90 Hz)
    gmma = np.tan(0.5*np.pi*min(90, nyq)/sampling_rate)
    gmma = ((1.0-gmma) / (1.0+gmma))

    # phase/amplitude estimator forgetting factor
    lambda_a = np.exp(np.log(0.05)/(W*sampling_rate+1))
    lambda_a = lambda_a*np.ones(num_harmonics+1, dtype=np.double)

    # initializing variables
    kappa_f, kappa_k = 0.0, np.zeros(num_harmonics+2, dtype=np.double)
    D, C = 10.0, 5.0
    f_n1, f_n2 = 0.0, 0.0

    # initializing the first oscillator
    u_kp = np.ones(num_harmonics, dtype=np.double)  # u_k
    u_k = np.ones(num_harmonics, dtype=np.double)  # u'_k

    # initializing the RLS parameters
    r1 = 100.0*np.ones(num_harmonics, dtype=np.double)
    r4 = 100.0*np.ones(num_harmonics, dtype=np.double)
    a = np.zeros(num_harmonics, dtype=np.double)
    b = np.zeros(num_harmonics, dtype=np.double)

    for n in range(N):
        # Lattice Filter
        f_n = x_f[n] + kappa_f*(1.0+alpha_f)*f_n1 - alpha_f*f_n2

        # Frequency Estimation
        C = lambda_f*C+(1-lambda_f)*f_n1*(f_n+f_n2)
        D = lambda_f*D+(1-lambda_f)*2.0*f_n1**2.0
        kappa_t = C/D
        if kappa_t < -1.0:
            kappa_t = -1.0
        if kappa_t > 1.0:
            kappa_t = 1.0
        kappa_f = gmma*kappa_f + (1.0-gmma)*kappa_t

        f_n2, f_n1 = f_n1, f_n  # Updating lattice states

        # Bandwidth and Forgetting Factor Updates
        alpha_f = alpha_st*alpha_f + (1-alpha_st)*alpha_inf
        lambda_f = lambda_st*lambda_f + (1-lambda_st)*lambda_inf

        # Discrete-Time Oscillators
        kappa_k[1], kappa_k[0] = 1.0, kappa_f
        e = data[n]

        for k in range(num_harmonics):
            # calculating Cos(kw) for k=1,2...
            kappa_k[k+2] = 2.0*kappa_f*kappa_k[k+1] - kappa_k[k]

            # Oscillator
            tmp = kappa_k[k+2]*(u_kp[k]+u_k[k])
            tmp2 = u_kp[k]
            u_kp[k] = tmp - u_k[k]
            u_k[k] = tmp + tmp2

            # Gain Control
            if (kappa_k[k+2]+1) == 0.0:  # 2015-08-23: Added to avoid zero-divison errors
                G = 1.0
            else:
                G = 1.5 - (u_kp[k]**2 - ((kappa_k[k+2]-1) / (kappa_k[k+2]+1)) * u_k[k]**2)
            if G < 0.0:
                G = 1.0

            u_kp[k] *= G
            u_k[k] *= G

        for k in range(num_harmonics):  # for each harmonic do:
            # Phase/Amplitude Adaptation
            sincmp = a[k]*u_k[k] + b[k]*u_kp[k]

            pli_data[n] += sincmp

            e = e - sincmp
            # --- Simplified RLS
            r1[k] = lambda_a[k]*r1[k] + u_k[k]**2.0
            r4[k] = lambda_a[k]*r4[k] + u_kp[k]**2.0
            a[k] = a[k] + u_k[k]*e/r1[k]
            b[k] = b[k] + u_kp[k]*e/r4[k]
            # ------

    return pli_data

_inner_loop_numba = numba.jit(nopython=True)(_inner_loop)
