import numpy as np


# TAKEN FROM: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!

    Args: 
        values: numpy.array with data
        quantiles: array-like with many quantiles needed
        sample_weight: array-like of the same length as `array`
        values_sorted: bool, if True, then will avoid sorting of
        initial array
        old_style: if True, will correct output to be consistent
        with numpy.percentile.
    Returns: 
        numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    # weighted_quantiles & values has to be 1d for this function to work
    return np.interp(quantiles, weighted_quantiles, values)

def median_sigma_from_samples(samples,weights=None):
    """Computes weighted median & 1-sigma bound from a set of samples w/ weights
    Args:
        samples:
        weights:
    
    Returns:
        median, 1sigma
    """


    if weights is not None:
        median,low,high = weighted_quantile(samples,[0.5,0.1586,0.8413],weights)
    else:
        median,low,high = np.quantile(samples,[0.5,0.1586,0.8413])
    sigma = ((high-median)+(median-low))/2

    return median,sigma
