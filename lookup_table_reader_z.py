from collections import namedtuple
import itertools
import numpy as np
import warnings

# for py2 py3 compatibility
from six.moves import range


class TableReader(object):

    def __init__(self, table_file):
        # table_file: [char] name of the lookup table, containing grid data
        # (e.g. CO intensity) as a function of grid coordinates
        # (metal,nH,sfr,colDen)

        # put this here so the file is only loaded once
        self.data = np.load(table_file)

        # defaults
        self.limitsMode = 'error'

    def _interpolatorQLinear(self, coords, data, point):
        # vectorized 5-D quntilinear interpolation over structured grid

        # Load the structured grid
        par1 = point[0]
        par10 = coords[0]
        par2 = point[1]
        par20 = coords[1]
        par3 = point[2]
        par30 = coords[2]
        par4 = point[3]
        par40 = coords[3]
        par5 = point[4]
        par50 = coords[4]

        varshape = par1.shape
        value21 = np.zeros(varshape)
        value20 = np.zeros(varshape)
        value31 = np.zeros(varshape)
        value30 = np.zeros(varshape)
        value41 = np.zeros(varshape)
        value40 = np.zeros(varshape)
        value51 = np.zeros(varshape)
        value50 = np.zeros(varshape)

        # Locate points
        index_par1max = par10.size - 2
        index_par2max = par20.size - 2
        index_par3max = par30.size - 2
        index_par4max = par40.size - 2
        index_par5max = par50.size - 2
        index1 = np.fmin(
                index_par1max, np.fmax(0, np.digitize(par1, par10) - 1)
                )
        index2 = np.fmin(
                index_par2max, np.fmax(0, np.digitize(par2, par20) - 1)
                )
        index3 = np.fmin(
                index_par3max, np.fmax(0, np.digitize(par3, par30) - 1)
                )
        index4 = np.fmin(
                index_par4max, np.fmax(0, np.digitize(par4, par40) - 1)
                )
        index5 = np.fmin(
                index_par5max, np.fmax(0, np.digitize(par5, par50) - 1)
                )

        # Interpolation
        for (p, q, w, v) in itertools.product([1, 2], repeat=4):
            indices = (p + index1 - 1, q + index2 - 1,
                       w + index3 - 1, v + index4 - 1)
            # over par5
            slope = (
                (data[indices + (index5 + 1,)] - data[indices + (index5,)])
                / (par50[index5 + 1] - par50[index5])
                )
            calc5 = ((par5 - par50[index5]) * slope) + data[indices+(index5,)]
            if v == 1:
                value50 = calc5
            else:
                value51 = calc5

            # over par4
            slope = (value51 - value50) / (par40[index4 + 1] - par40[index4])
            calc4 = (par4 - par40[index4]) * slope + value50
            if w == 1:
                value40 = calc4
            else:
                value41 = calc4
            # over par3
            slope = (value41 - value40) / (par30[index3 + 1] - par30[index3])
            calc3 = (par3 - par30[index3]) * slope + value40
            if q == 1:
                value30 = calc3
            else:
                value31 = calc3
            # over par2
            slope = (value31 - value30) / (par20[index2 + 1] - par20[index2])
            calc2 = (par2 - par20[index2]) * slope + value30
            if p == 1:
                value20 = calc2
            else:
                value21 = calc2
        # over par1
        slope = (value21 - value20) / (par10[index1 + 1] - par10[index1])
        value = (par1 - par10[index1]) * slope + value20

        return value

    @property
    def limitsMode(self):
        '''
        Method [str]: error -- raises error
                      clip -- caps values to the limits of coords_str
                      value -- fills in a constant value
                      leave -- do nothing
        Fill [float]: the value to use when Method == 'value'
        '''

        return self.__limitsMode

    @limitsMode.setter
    def limitsMode(self, value):

        try:
            method, fill = value
        except (TypeError, ValueError):
            method = value
            fill = None

        method = method.lower()

        self.__limitsMode = namedtuple(
                'Limit', ['Method', 'Fill']
                )(method, fill)

    @property
    def copyPoints(self):
        '''
        True -- arrays provided in getValues are not edited if element values
                must be edited (i.e. in _check_limits)
        False -- the original array is edited
        '''
        return self.__copyPoints

    @copyPoints.setter
    def copyPoints(self, value):
        self.__copyPoints = value

    def _check_limits(self, points, coords, coords_name):
        '''
        Args:
            points [numpy.array]: array to be tested
            coords [numpy.array]: defined values
            coords_name [str]: The name of the property
                               (for error/warning output)
            copy [bool]: True -- returns new array with updated values
                         False -- updates the original array in place
        '''

        min_value = np.nanmin(coords)
        max_value = np.nanmax(coords)

        outside_limits = np.logical_or(
                points < min_value,
                points > max_value,
                )

        if outside_limits.any():
            message = 'Limits exceeded: {:s}'.format(coords_name)

            if self.limitsMode.Method == 'error':
                raise ValueError(message)

            elif self.limitsMode.Method == 'leave':
                warnings.warn(message)

            else:
                warnings.warn(message)

                if self.copyPoints:
                    # to avoid updating original input values
                    points = points.copy()

                if self.limitsMode.Method == 'clip':
                    points.clip(min=min_value, max=max_value, out=points)

                elif self.limitsMode.Method == 'value':
                    points[outside_limits] = self.limitsMode.Fill

                else:
                    raise ValueError('Unknown method')

        return points

    def getValues(self, redshift_points, column_points,
                  metal_points, nh_points, sfr_points,
                  intensity=False, log_input=False, log_output=False,
                  ):
        '''
         Args:
             redshift, column_points, metal_points, nh_points, sfr_points:
             [ndarray] must be on linear scale!

             log_input/log_output:
               - False: input grid coordinates remain on linear scale/grid data
                         remain on linear scale
               - True: input grid coordinates are transformed to log scale
                        (except metallicity)/grid data are transformed to log
                        scale (e.g. CO intensity -> log(CO intensity))

         Returns:
         dict[str, np.ndarray]
             line intensity: 'co', 'ci', 'cii'
             abundance: 'h2', 'hi'
        '''

        # input must be same length, check before calculations
        same_length = (
                redshift_points.shape == column_points.shape ==
                metal_points.shape == nh_points.shape == sfr_points.shape
                )
        assert same_length, 'Not all input array are the same shape'

        # -- coords arrays ---------------------------------------------------
        redshift = self.data['redshift']
        column_density = self.data['column_density']
        metalgrid = self.data['metalgrid']
        nhgrid = self.data['nhgrid']
        sfrgrid = self.data['sfrgrid']

        # -- check limits ----------------------------------------------------
        redshift_points = self._check_limits(
                redshift_points, redshift, 'Redshift',
                )
        column_points = self._check_limits(
                column_points, column_density, 'Column Density',
                )
        metal_points = self._check_limits(
                metal_points, metalgrid, 'Metallicity',
                )
        nh_points = self._check_limits(
                nh_points, nhgrid, 'H Number Density (volume)',
                )
        sfr_points = self._check_limits(
                sfr_points, sfrgrid, 'Star Formation Rate',
                )

        # -- data arrays -----------------------------------------------------
        data_arrays = {}
        if intensity:
            data_arrays["co"] = self.data['CO_intTB_array']
            data_arrays["ci"] = self.data['CI_intTB_array']
            data_arrays["cii"] = self.data['CII_intTB_array']

        else:
            data_arrays["co"] = self.data['CO_lines_array']
            data_arrays["ci"] = self.data['CI_lines_array']
            data_arrays["cii"] = self.data['CII_lines_array']

        data_arrays["h2"] = self.data['H2_abu_array']
        data_arrays["hi"] = self.data['HI_abu_array']

        if log_output:
            # Suppress: RuntimeWarning: invalid value encountered in log10
            with np.errstate(invalid='ignore'):
                # do this before nan_to_num as all values that are
                # <= 0 will also become 'bad' values
                for key in data_arrays:
                    data_arrays[key] = np.log10(data_arrays[key])

        # Convert nan and inf to numbers
        for key in data_arrays:
            data_arrays[key] = np.nan_to_num(data_arrays[key])

        if log_input:
            point = (redshift_points, metal_points, np.log10(column_points),
                     np.log10(nh_points), np.log10(sfr_points))
            coords = (redshift, metalgrid, np.log10(column_density),
                      np.log10(nhgrid), np.log10(sfrgrid))
        else:
            point = (redshift_points, metal_points, column_points,
                     nh_points, sfr_points)
            coords = (redshift, metalgrid, column_density, nhgrid, sfrgrid)

        output = {}
        output["co"] = np.array([
                self._interpolatorQLinear(
                        coords,
                        data_arrays["co"][:, :, :, :, :, r],
                        point
                        )
                for r in range(10)
                ])
        output["ci"] = np.array([
                self._interpolatorQLinear(
                        coords,
                        data_arrays["ci"][:, :, :, :, :, r],
                        point
                        )
                for r in range(2)
                ])
        output["cii"] = self._interpolatorQLinear(
                coords, data_arrays["cii"], point
                )
        output["h2"] = self._interpolatorQLinear(
                coords, data_arrays["h2"], point
                )
        output["hi"] = self._interpolatorQLinear(
                coords, data_arrays["hi"], point
                )

        if log_output:
            for key in output:
                # Array updated in place
                np.power(10., output[key], out=output[key])

        output["h2"] *= 2.

        return output
