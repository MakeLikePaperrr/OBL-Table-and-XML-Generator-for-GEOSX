
from __future__ import division, print_function, absolute_import
from src.cubic import *
from src.cubicvt import *
from src.cubicprw import *


def preos(mix_or_component, mixrule = 'qmr' ,volume_translation = False, water=False):
    '''
    Peng Robinson EoS

    Parameters
    ----------
    mix_or_component : object
        created with component or mixture, in case of mixture object has to
        have interactions parameters.
    mixrule : str
        available opitions 'qmr', 'mhv_nrtl', 'mhv_unifac', 'mhv_rk',
        'mhv_wilson', 'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac'.
    volume_translation: bool
        If True, the volume translated version of this EoS will be used.

    Returns
    -------
    eos : object
        eos used for phase equilibrium calculations
    '''

    if volume_translation:
        eos = vtprmix(mix_or_component, mixrule)
    else:
        if water:
            eos = prw(mix_or_component, mixrule)
        else:
            eos = prmix(mix_or_component, mixrule)

    return eos

