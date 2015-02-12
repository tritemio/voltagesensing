"""
This module defines a container class for patch-clamp experiments.
"""

from io import open
import os
import numpy as np
import pandas as pd


class PatchDataset:
    _DAQ_Sq = 'Sqrwave_DAQ.txt'
    _DAQ_Tri = 'Triwave_DAQ.txt'
    _Cam_Sq = 'Sq_camera.bin'
    _Cam_Tri = 'Tri_camera.bin'
    _ParameterFile = 'experimental_parameters.txt'
    _drop_frames = 4   # Initial frames containing garbage

    def __init__(self, folder):
        assert os.path.isdir(folder), "Folder not found."
        self.folder = os.path.abspath(folder) + os.path.sep

    def _read_params(self):
        fname = self.folder + self._ParameterFile
        params = {}
        with open(fname, 'r') as f:
            for line in f:
                key, value = line.split(':')
                params[key.strip()] = float(value.replace('_', ''))
        return params

    @property
    def params(self):
        if not hasattr(self, '_params'):
            self._params = self._read_params()
        return self._params

    def _read_video(self):
        fname = self.folder + self._Cam_Sq
        raw_data = np.fromfile(fname, dtype=np.uint16)
        ncols = self.params['Horizontal pixel']
        nrows = self.params['Vertical pixel']
        nframes = raw_data.size / (nrows * ncols)
        assert raw_data.size == nframes*nrows*ncols, \
            "Array size (%d) not consistent with shape (%d, %d, %d)" % \
            (raw_data.size, nframes, nrows, ncols)
        return raw_data.reshape(nframes, nrows, ncols)[self._drop_frames:]

    @property
    def video(self):
        if not hasattr(self, '_video'):
            self._video = self._read_video()
        return self._video

    def _read_daq(self):
        fname = self.folder + self._DAQ_Sq
        with open(fname, 'r') as f:
            head = f.readline()
        daq = pd.DataFrame.from_csv(fname, sep='\t')
        daq = daq.drop(daq.columns[-1], axis=1)
        daq.columns = (col.strip() for col in head.split('\t')[1:]
                                   if len(col) > 0)
        return daq

    @property
    def daq(self):
        if not hasattr(self, '_daq'):
            self._daq = self._read_daq()
        return self._daq

    @property
    def period(self):
        """Number of camera frames per square-wave period.
        """
        if not hasattr(self, '_period'):
            period = self.params['Square frame rate (Hz)'] / \
                     self.params['Square wave frequency (Hz)']
            assert int(period) == period, \
                "Alternation period should be integer (is %.2f)" % period
            self._period = int(period)
        return self._period

    @property
    def camera_rate(self):
        """Camera frame rate in Hz."""
        return self.params['Square frame rate (Hz)']

    @property
    def daq_rate(self):
        """DAQ acquisition rate in Hz."""
        return self.params['Square wave DAQ rate (Hz)']

    @property
    def time(self):
        """Time axis sampled at the camera frame rate."""
        if not hasattr(self, '_time'):
            self._time = np.arange(self.video.shape[0])/self.camera_rate
        return self._time

    def _read_dac_coldata(self, name):
        """Read the column `name` from self.daq downsampling to camera rate.
        """
        decimate = self.daq_rate/self.camera_rate
        coldata = np.array(self.daq[name])
        coldata = coldata.reshape(coldata.size/decimate, decimate).mean(1)
        return coldata[self._drop_frames:]

    @property
    def voltage(self):
        """Voltage sampled at the camera frame rate."""
        if not hasattr(self, '_voltage'):
            self._voltage = self._read_dac_coldata('AI V_m')
        return self._voltage

    @property
    def current(self):
        """Current sampled at the camera frame rate."""
        if not hasattr(self, '_current'):
            self._current = self._read_dac_coldata('AI scaled')
        return self._current

