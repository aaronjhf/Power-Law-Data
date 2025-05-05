import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
import contextlib




def silverstein_density(
      evals: np.ndarray, 
      alpha: float, 
      q: float, 
      Lc: float, 
      eps: float = 1e-9
      )-> np.ndarray:
    """
    Compute the limiting spectral density ρ(z) for a power‐law prior H(t) ∝ t^{-1/α − 1}
    on [Lc, 1], via the Silverstein equation.

    Parameters
    ----------
    evals
        1D array of real evaluation points z_i where ρ(z_i) is desired. Should be fed in ordered largest to smallest for best control.
    alpha
        Power‐law exponent parameter α > 0.
    q
        Aspect‐ratio parameter (p/n).
    Lc
        Lower cutoff of the power‐law support, 0 < Lc < 1.
    eps
        Small imaginary part added to z to regularize the root‐finder.

    Returns
    -------
    rho
        1D array of the same length as `evals`, containing ρ(z_i).
    """

    c = 1/alpha*1/(1-np.power(Lc, 1/alpha))

    def diff(q, z, mf):
        delta = (-1 + q - q*z*mf)

        int_re = lambda t: np.real(c*np.exp(t+t/alpha)/( mf*(Lc*delta + z*np.exp(t)) ))
        int_im = lambda t: np.imag(c*np.exp(t+t/alpha)/( mf*(Lc*delta + z*np.exp(t) ) ))
        diff_re, _ = quad(int_re, np.log(Lc), 0)
        diff_im, _ = quad(int_im, np.log(Lc), 0)

        return [1-diff_re, diff_im]

    def get_rho(q, z, mf0 = None):

        def complex_zero(mflist):
            return diff(q, z, mflist[0]+1j*mflist[1])


        with contextlib.redirect_stdout(None):
            result = root(complex_zero, [1/np.real(z), 1/np.real(z)] if mf0 is None else mf0, method='hybr')


        if result.success:
            return result.x[1]/np.pi, [result.x[0], result.x[1]]
        else:
            return None, None
            print("Root finding failed:", result.message)
    
    spec_array = []
    rho, mf0 = get_rho(q, evals[0]+1j*eps)
    spec_array.append(rho)

    for z in evals[1:]:
        rho, mf0 = get_rho(q, z+1j*eps, mf0)
        spec_array.append(rho)
    return np.array(spec_array, dtype=float)


def end_pts(spec_dict):
  deriv_dict = {}
  sorted_keys = sorted(list(spec_dict.keys()))
  for i in range(len(sorted_keys)-1):
    z0 = sorted_keys[i]
    if spec_dict[z0] is None:
      continue
    z1 = [z for z in sorted_keys[i+1:] if spec_dict[z] is not None][0]
    deriv_dict[z0] = (np.log(abs(spec_dict[z1])/abs(spec_dict[z0])))

  #print(deriv_dict)
  z_L = max([z for z in sorted_keys[:len(sorted_keys)//2] if spec_dict[z] is not None], key = lambda z: deriv_dict[z])
  z_R = min([z for z in sorted_keys[len(sorted_keys)//2:-1] if spec_dict[z] is not None], key = lambda z: deriv_dict[z])

  return z_L, z_R


