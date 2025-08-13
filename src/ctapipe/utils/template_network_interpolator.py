import gzip
import os
import pickle
import re

import numpy as np
import numpy.ma as ma
import tensorflow as tf

from .unstructured_interpolator import UnstructuredInterpolator


class BaseTemplate:
    """
    Base class for template interpolators, cheifly associated with code for the special
    interpolation performed on the zenith and azimuth directions. These dimensions are
    treated separately as they should be in principle generated on a grid, so Delauny
    triangulation based interpolation becomes needlessly inefficient
    """

    def __init__(self):
        self.zeniths = None
        self.azimuths = None
        self.interpolator = None
        self.keys = None
        self.values = None
        self.bounds = None

    def reset(self):
        """
        Reset method to delete some saved results from the previous event
        """

        for i in self.interpolator:
            for j in i:
                if j is not None:
                    j.reset()

    def _create_table_matrix(self, keys, values):
        """
        Create the array of interpolators to be used for all combinations of zenith and
        azimuth included in the templates
        Parameters
        ----------
        keys: ndarray
            Template grid points
        values: ndarray
            Template values
        Returns
        -------
        None
        """

        # First lets store the unique zeniths and azimuths stored in our table
        zeniths = np.sort(np.unique(keys.T[0]))
        azimuths = np.sort(np.unique(keys.T[1]))
        # Assuming these are created on a grid create an array to hold the unstructured
        # interpolator objects for each zenith and azimuth
        # Don't create any yet though as they are slow!
        interpolator = np.empty((len(zeniths), len(azimuths)), dtype=object)

        self.interpolator = interpolator
        self.zeniths = zeniths
        self.azimuths = azimuths

        self.keys = keys
        self.values = values

    def _get_bounds(self, zenith, azimuth):
        """
        Get bounding indices of a given direction in the zenith, azimuth grid
        This code is ugly and not easily transferable, but should be fast enough
        Parameters
        ----------
        zenith: float
            Zenith angle (degrees)
        azimuth: float
            Azimuth angle (degrees)
        Returns
        -------
        tuple: (zenith indices, azimuth indices)
        """

        # If we only have one zenith angle available in the templates, don't bother
        # searching
        if len(self.zeniths) == 0:
            zenith_bound = self.zeniths[0], self.zeniths[0]
        else:
            # Otherwise search for where our zenith lies in the available range
            index = np.searchsorted(self.zeniths, zenith, side="left")

            index_upper = index
            # Unless we are the edge of the range the boundary should be above this value
            if index != len(self.zeniths) - 1 and index != 0:
                index_upper += 1

            # If we are not at the edge we need to reduce the index by one
            if index != 0:
                index -= 1
                index_upper -= 1

            zenith_bound = (index, index_upper)

        # Do the same again for azimuth angle
        if len(self.azimuths) == 0:
            azimuth_bound = self.azimuths[0], self.azimuths[0]
        else:
            index = np.searchsorted(self.azimuths, azimuth)
            # Except in this case we need to loop back around rather than stay at the edge
            if index != 0:
                index -= 1

            index_upper = index
            if index != len(self.azimuths) - 1:
                index_upper += 1
            else:
                index_upper = 0

            azimuth_bound = (index, index_upper)

        # Return our boundaries
        return zenith_bound, azimuth_bound

    def _create_interpolator(self, zenith_bin, azimuth_bin):
        """
        Creates unstructured interpolator object in a given zenith and azimuth bin when
        needed
        Parameters
        ----------
        zenith_bin: int
            Bin number of requested zenith
        azimuth_bin: int
            Bin number of requested azimuth
        Returns
        -------
        None
        """
        # Get our requested zenith and azimuth

        zenith = self.zeniths[zenith_bin]
        azimuth = self.azimuths[azimuth_bin]

        # Select these values from our range of keys
        selection = np.logical_and(self.keys.T[0] == zenith, self.keys.T[1] == azimuth)

        # Create interpolator using this selection
        # Currently impact is not set up for offset dependent templates.
        # Therefore remove offset (last) dimension from interpolator
        self.interpolator[zenith_bin][azimuth_bin] = UnstructuredInterpolator(
            self.keys[selection].T[2:5].T,
            self.values[selection],
            remember_last=False,
            bounds=self.bounds,
            dtype="float32",
        )

        # We can now remove these entries.
        self.keys = self.keys[np.invert(selection)]
        self.values = self.values[np.invert(selection)]

    def perform_interpolation(self, zenith, azimuth, interpolation_array, points):
        """
        Interpolate values for given zenith and azimuth using bilinear interpolation.
        """
        zenith_bounds, azimuth_bounds = self._get_bounds(zenith, azimuth)
        zl, zu = zenith_bounds
        al, au = azimuth_bounds

        def eval(z, a):
            # Evaluate or create the interpolator for the given zenith and azimuth bin
            return self._evaluate_interpolator(z, a, interpolation_array, points)

        v_ll = eval(zl, al)
        if len(self.zeniths) == 1 and len(self.azimuths) == 1:
            # If only one zenith and azimuth, return directly
            return v_ll

        v_ul = eval(zu, al)
        v_lu = eval(zl, au)
        v_uu = eval(zu, au)

        # Interpolate along zenith for lower and upper azimuth bounds
        v_l = self._linear_interpolation(
            zenith, self.zeniths[zl], self.zeniths[zu], v_ll, v_ul
        )
        v_u = self._linear_interpolation(
            zenith, self.zeniths[zl], self.zeniths[zu], v_lu, v_uu
        )
        # Interpolate along azimuth between the two zenith-interpolated values
        return self._linear_interpolation(
            azimuth, self.azimuths[al], self.azimuths[au], v_l, v_u
        )

    def _evaluate_interpolator(
        self, zenith_bin, azimuth_bin, interpolation_array, points
    ):
        """
        Helper function to create and evaluate the interpolator for a given zenith and azimuth bin.
        If the interpolator does not exist, it is created.
        """
        if self.interpolator[zenith_bin][azimuth_bin] is None:
            self._create_interpolator(zenith_bin, azimuth_bin)

        return self.interpolator[zenith_bin][azimuth_bin](interpolation_array, points)

    @staticmethod
    def _linear_interpolation(point, grid1, grid2, value1, value2):
        """
        Simple function to perform linear interpolation between two values
        Parameters
        ----------
        point: float
            Point at which to perform interpolation
        grid1: float
            Lower interpolation point
        grid2: float
            Upper interpolation point
        value1: ndarray
            Lower interpolation value
        value2: ndarray
            Upper interpolation value
        Returns
        -------
        ndarray: Interpolated values
        """
        if np.abs(grid1 - grid2) < 1e-10:
            return value1

        result = ((value2 - value1) * (point - grid1) / (grid2 - grid1)) + value1

        return result


class TemplateNetworkInterpolator(BaseTemplate):
    """
    Class for interpolating between the the predictions
    """

    def __init__(self, template_file, bounds=((-5, 1), (-1.5, 1.5))):
        """
        Parameters
        ----------
        template_file: str
            Location of pickle file containing ImPACT NN templates
        """

        super().__init__()

        self.template_file = template_file

        input_dict = None
        with gzip.open(template_file, "r") as file_list:
            input_dict = pickle.load(file_list)

        # The dict has at its highest level two keys: data and tel_type.
        # The first just contains the template dict, the second a string to validate the tel_type
        # that the templates are made for.

        data_input_dict = input_dict["data"]
        self.tel_type_string = input_dict["tel_type"]

        keys = np.array(list(data_input_dict.keys()))
        values = np.array(list(data_input_dict.values()), dtype=np.float32)
        self.no_zenaz = False
        self.bounds = bounds

        # First check if we even have a zen and azimuth entry
        if len(keys[0]) > 4:
            # If we do then for the sake of speed lets
            self._create_table_matrix(keys, values)
        else:
            # If not we work as before
            # Currently impact is not set up for offset dependent templates.
            # Therefore remove offset (last) dimension from interpolator
            self.interpolator = UnstructuredInterpolator(
                keys[:-1], values, remember_last=False, bounds=bounds
            )
            self.no_zenaz = True

    def __call__(self, zenith, azimuth, energy, impact, xmax, xb, yb):
        """
        Evaluate interpolated templates for a set of shower parameters and pixel positions
        Parameters
        ----------
        energy: array-like
            Energy of interpolated template
        impact: array-like
            Impact distance of interpolated template
        xmax: array-like
            Depth of maximum of interpolated templates
        xb: array-like
            Pixel X position at which to evaluate template
        yb: array-like
            Pixel X position at which to evaluate template
        Returns
        -------
        ndarray: Pixel amplitude expectation values
        """
        array = np.stack((energy, impact, xmax), axis=-1)
        points = ma.dstack((xb, yb))

        if self.no_zenaz:
            interpolated_value = self.interpolator(array, points)
        else:
            interpolated_value = self.perform_interpolation(
                zenith, azimuth, array, points
            )

        interpolated_value[interpolated_value < 0] = 0

        return interpolated_value

    # Two interpolators are the same if they are generated from the same file
    def __eq__(self, other):
        return self.template_file == other.template_file


class TimeGradientInterpolator(BaseTemplate):
    """
    Class for interpolating between the time gradient predictions
    """

    def __init__(self, template_file):
        """
        Parameters
        ----------
        template_file: str
            Location of pickle file containing ImPACT NN templates
        """

        super().__init__()

        self.template_file = template_file

        input_dict = None
        with gzip.open(template_file, "r") as file_list:
            input_dict = pickle.load(file_list)

        # The dict has at its highest level two keys: data and tel_type.
        # The first just contains the template dict, the second a string to validate the tel_type
        # that the templates are made for.

        data_input_dict = input_dict["data"]
        self.tel_type_string = input_dict["tel_type"]

        keys = np.array(list(data_input_dict.keys()))
        values = np.array(list(data_input_dict.values()), dtype=np.float32)
        self.no_zenaz = False

        # First check if we even have a zen and azimuth entry
        if len(keys[0]) > 4:
            # If we do then for the sake of speed lets
            self._create_table_matrix(keys, values)
        else:
            # If not we work as before
            # Currently impact is not set up for offset dependent templates.
            # Therefore remove offset (last) dimension from interpolator
            self.interpolator = UnstructuredInterpolator(
                keys[:-1], values, remember_last=False
            )
            self.no_zenaz = True

    def __call__(self, zenith, azimuth, energy, impact, xmax):
        """
        Evaluate expected time gradient for a set of shower parameters and pixel positions
        Parameters
        ----------
        energy: array-like
            Energy of interpolated template
        impact: array-like
            Impact distance of interpolated template
        xmax: array-like
            Depth of maximum of interpolated templates
        Returns
        -------
        ndarray: Time Gradient expectation and RMS values
        """
        array = np.stack((energy, impact, xmax), axis=-1)

        if self.no_zenaz:
            interpolated_value = self.interpolator(array, None)
        else:
            interpolated_value = self.perform_interpolation(
                zenith, azimuth, array, None
            )

        return interpolated_value

    # Two interpolators are the same if they are generated from the same file
    def __eq__(self, other):
        return self.template_file == other.template_file


def load_prediction_files_filtered(directory):
    """
    Finds all files in the given directory matching the pattern:
    predict_{telescope}_{zenith}deg_{azimuth}azm_{offset}off_{species}.keras
    Only includes files where telescope and species match the specified values.

    Returns a dictionary where the key is a tuple of (zenith, azimuth, offset)
    and the value is the absolute file path.
    """
    pattern = re.compile(
        r"predict_(?P<telescope>[^_]+)_(?P<zenith>\d+)deg_(?P<azimuth>\d+)azm_(?P<offset>[\d\.]+)off_(?P<species>[^\.]+)\.keras"
    )
    result = {}
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            key = (
                int(match.group("zenith")),
                int(match.group("azimuth")),
                float(match.group("offset")),
            )
            abs_path = os.path.abspath(os.path.join(directory, filename))
            model = tf.keras.models.load_model(abs_path)
            model.layers[-1].activation = tf.keras.activations.linear

            result[key] = model
    return result


def custom_symlog(value, linear_threshold=10.0):
    return np.where(
        np.abs(value) < linear_threshold,
        value,
        np.sign(value)
        * (np.emath.logn(2, np.abs(value / linear_threshold)) + linear_threshold),
    )


class FreePACTInterpolator(BaseTemplate):
    """
    Class for interpolating between the FreePACT predictions
    """

    def __init__(self, directory):
        """
        Parameters
        ----------
        directory: str
            Directory containing the FreePACT prediction files
        """

        super().__init__()

        data_input_dict = load_prediction_files_filtered(directory)
        # self.tel_type_string = telescope_type

        keys = np.array(list(data_input_dict.keys()))
        values = list(data_input_dict.values())
        self.no_zenaz = False

        # First check if we even have a zen and azimuth entry
        if len(keys[0]) > 4:
            # If we do then for the sake of speed lets
            self._create_table_matrix(keys, values)
        else:
            # If not we work as before
            # Currently impact is not set up for offset dependent templates.
            # Therefore remove offset (last) dimension from interpolator
            self.interpolator = values[0]
            self.no_zenaz = True

    def _create_interpolator(self, zenith_bin, azimuth_bin):
        """
        Creates unstructured interpolator object in a given zenith and azimuth bin when
        needed
        Parameters
        ----------
        zenith_bin: int
            Bin number of requested zenith
        azimuth_bin: int
            Bin number of requested azimuth
        Returns
        -------
        None
        """
        # Get our requested zenith and azimuth

        zenith = self.zeniths[zenith_bin]
        azimuth = self.azimuths[azimuth_bin]

        # Select these values from our range of keys
        selection = np.logical_and(self.keys.T[0] == zenith, self.keys.T[1] == azimuth)

        # Create interpolator using this selection
        # Currently freepact is not set up for offset dependent templates.
        # Therefore remove offset (last) dimension from interpolator
        self.interpolator[zenith_bin][azimuth_bin] = self.values[selection]

        # We can now remove these entries.
        self.keys = self.keys[np.invert(selection)]
        self.values = self.values[np.invert(selection)]

    def _evaluate_interpolator(self, zenith_bin, azimuth_bin, interpolation_array):
        """
        Helper function to create and evaluate the interpolator for a given zenith and azimuth bin.
        If the interpolator does not exist, it is created.
        """
        if self.interpolator[zenith_bin][azimuth_bin] is None:
            self._create_interpolator(zenith_bin, azimuth_bin)

        return self.interpolator[zenith_bin][azimuth_bin].predict(interpolation_array)

    def __call__(self, zenith, azimuth, energy, impact, xmax, xb, yb, amplitude):
        """
        Evaluate expected time gradient for a set of shower parameters and pixel positions
        Parameters
        ----------
        energy: array-like
            Energy of interpolated template
        impact: array-like
            Impact distance of interpolated template
        xmax: array-like
            Depth of maximum of interpolated templates
        Returns
        -------
        ndarray: Time Gradient expectation and RMS values
        """
        shape = xb.shape
        repeat_num = xb.shape[-1]

        amplitude = custom_symlog(amplitude / 10, linear_threshold=2)
        impact = np.log10(impact)
        energy = np.log10(energy)
        xmax = np.log10(xmax)

        array = np.stack(
            (
                xb.ravel(),
                yb.ravel(),
                np.repeat(impact, repeat_num, axis=-1),
                np.repeat(energy, repeat_num, axis=-1),
                np.repeat(xmax, repeat_num, axis=-1),
                amplitude.ravel(),
            ),
            axis=-1,
        )

        if self.no_zenaz:
            interpolated_value = self.interpolator.predict(array)
            interpolated_value = interpolated_value.reshape(shape)
        else:
            interpolated_value = self.perform_interpolation(zenith, azimuth, array)

        return interpolated_value


class DummyTemplateInterpolator:
    def __call__(self, zenith, azimuth, energy, impact, xmax, xb, yb):
        return np.ones_like(xb)

    def reset(self):
        return True


class DummyTimeInterpolator:
    def __call__(self, energy, impact, xmax):
        return np.ones_like(energy)

    def reset(self):
        return True
