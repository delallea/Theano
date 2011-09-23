#!/usr/bin/env python

"""
Display information on numpy casting of mixed scalar / array operations.
"""

__authors__   = "Olivier Delalleau"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "Olivier Delalleau <delallea@iro>"


import operator, sys

import numpy
import theano
from theano import tensor


def main():

    type_combinations = [
            ('int8', 'int32'),
            ]

    values = {
            'int8': [0, 2**7 - 1, -2**7],
            'int32': [0, 1, -1, 2**7 - 1, -2**7, 2**7, -2**7 - 1],
            }

    for op in (operator.add, ):
        for array_type, scalar_type in type_combinations:
            for array_val in values[array_type]:
                for scalar_val in values[scalar_type]:
                    array = numpy.array([array_val] * 2, dtype=array_type)
                    scalar = numpy.array(scalar_val, dtype=scalar_type)
                    output = op(array, scalar)
                    print '%s: %s (%s) / %s (%s) -> %s (%s)' % (
                            op, array, array_type, scalar, scalar_type,
                            output, output.dtype)

    return 0

if __name__ == '__main__':
    sys.exit(main())
