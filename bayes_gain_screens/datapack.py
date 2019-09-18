import tables as tb
from . import logging
import os
import numpy as np
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import sys
import itertools
import re

from . import deprecated



def _load_array_file(array_file):
    '''Loads a csv where each row is x,y,z in geocentric ITRS coords of the antennas'''

    try:
        types = np.dtype({'names': ['X', 'Y', 'Z', 'diameter', 'station_label'],
                          'formats': [np.double, np.double, np.double, np.double, 'S16']})
        d = np.genfromtxt(array_file, comments='#', dtype=types)
        diameters = d['diameter']
        labels = np.array(d['station_label'].astype(str))
        locs = ac.SkyCoord(x=d['X'] * au.m, y=d['Y'] * au.m, z=d['Z'] * au.m, frame='itrs')
        Nantenna = int(np.size(d['X']))
    except:
        d = np.genfromtxt(array_file, comments='#', usecols=(0, 1, 2))
        locs = ac.SkyCoord(x=d[:, 0] * au.m, y=d[:, 1] * au.m, z=d[:, 2] * au.m, frame='itrs')
        Nantenna = d.shape[0]
        labels = np.array([b"ant{:02d}".format(i) for i in range(Nantenna)])
        diameters = None
    return np.array(labels).astype(np.str_), locs.cartesian.xyz.to(au.m).value.transpose()




class DataPack(object):
    # _H: tb.File
    _arrays = os.path.dirname(sys.modules["bayes_filter"].__file__)
    lofar_array = os.path.join(_arrays, 'arrays/lofar.hba.antenna.cfg')
    lofar_cycle0_array = os.path.join(_arrays, 'arrays/lofar.cycle0.hba.antenna.cfg')
    gmrt_array = os.path.join(_arrays, 'arrays/gmrtPos.csv')

    def __init__(self, filename, readonly=False):
        if isinstance(filename, DataPack):
            filename = filename.filename
        self.filename = os.path.abspath(filename)
        self.readonly = readonly
        self._H = None
        self._contexts_open = 0
        self._selection = None
        self._current_solset = None
        self.axes_order = ['pol','dir','ant','freq','time']
        self.axes_atoms = {'pol':(np.str_,tb.StringAtom(16)),
                           'dir':(np.str_,tb.StringAtom(128)),
                           'ant':(np.str_,tb.StringAtom(16)),
                           'freq':(np.float64,tb.Float64Atom()),
                           'time':(np.float64,tb.Float64Atom())}

    @property
    def axes_order(self):
        return self._axes_order

    @axes_order.setter
    def axes_order(self, axes):
        if not isinstance(axes,(tuple, list)):
            raise ValueError("axes should be a list or tuple. {}".format(type(axes)))
        order = []
        for axis in axes:
            if axis not in ['ant','dir','freq','pol','time']:
                raise ValueError("Axis {} not a valid axis.".format(axis))
            if axis in order:
                raise ValueError("Found duplicate in ordering. {}".format(axes))
            order.append(axis)
        self._axes_order = order

    @property
    def readonly(self):
        return self._readonly

    @readonly.setter
    def readonly(self, value):
        if not isinstance(value, bool):
            raise ValueError("Readonly must be a bool.")
        self._readonly = value

    def __enter__(self):

        if self._contexts_open == 0:
            self._H = tb.open_file(self.filename, mode='r' if self.readonly else 'a')
        self._contexts_open += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._contexts_open == 1:
            self._H.close()
            self._H = None
        self._contexts_open -= 1


    @property
    def current_solset(self):
        return self._current_solset

    @current_solset.setter
    def current_solset(self, solset):
        if solset not in self.solsets:
            raise ValueError("Solset {} does not exist.".format(solset))
        self._current_solset = solset

    @property
    def solsets(self):
        with self:
            return [k for k,v in self._H.root._v_groups.items()]

    @property
    def soltabs(self):
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]
            return [k for k, v in solset_group._v_groups.items()]


    @deprecated("Use add_solset")
    def switch_solset(self,solset, antenna_labels=None, antennas=None, array_file=None, directions=None,
                      patch_names=None):
        self.add_solset(solset, antenna_labels, antennas, array_file, directions,
                      patch_names)

    def add_solset(self,solset, antenna_labels = None, antennas=None, array_file=None, patch_names=None, directions=None):
        if solset in self.solsets:
            logging.warn("Solset {} already exists.".format(solset))
            self.current_solset = solset
            return
        with self:
            self._H.create_group(self._H.root, solset, title='Solset: {}'.format(solset))
            self.current_solset = solset
            self.add_antenna_table()
            if antennas is None:
                antenna_labels, antennas = _load_array_file(self.lofar_array if array_file is None else array_file)
            self.set_antennas(antenna_labels, antennas)
            self.add_directions_table()
            if directions is not None:
                self.set_directions(patch_names, directions)
            logging.info("Created solset {}.".format(solset))

    def add_soltab(self,soltab, values=None, weights=None, weightDtype='f16', **axes):
        if soltab in self.soltabs:
            logging.warn('Soltab {} already exists.'.format(soltab))
            return
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]
            self._H.create_group(solset_group,soltab, "Soltab: {}".format(soltab))
            soltab_group = solset_group._v_groups[soltab]
            soltab_group._v_attrs['parmdb_type'] = ""
            shape = []
            ordered_axes = []
            for axis_name in self.axes_order:
                if axis_name not in axes.keys():
                    logging.info("Soltab missing axis {}".format(axis_name))
                    continue
                shape.append(len(axes[axis_name]))
                ordered_axes.append(axis_name)
                self._H.create_array(soltab_group, axis_name,obj=np.array(axes[axis_name]),title='Axis: {}'.format(axis_name))#,atom=self.axes_atoms[axis_name][1])
            if values is None:
                values = np.zeros(shape)
            self._H.create_array(soltab_group, 'val', obj=values.astype(np.float64), atom=tb.Float64Atom())
            val_leaf = soltab_group._v_leaves['val']
            val_leaf.attrs['AXES'] = ','.join(ordered_axes)
            if weightDtype not in ['f16', 'f32', 'f64']:
                raise ValueError("Allowed weight dtypes are 'f16','f32', 'f64'")
            if weights is None:
                weights = np.ones(shape)
            if weightDtype == 'f16':
                self._H.create_array(soltab_group, 'weight', obj=weights.astype(np.float16), title='Weights',atom=tb.Float16Atom())
            elif weightDtype == 'f32':
                self._H.create_array(soltab_group, 'weight', obj=weights.astype(np.float32), title='Weights',atom=tb.Float32Atom())
            elif weightDtype == 'f64':
                self._H.create_array(soltab_group, 'weight', obj=weights.astype(np.float64), title='Weights',atom=tb.Float64Atom())
            weight_leaf = soltab_group._v_leaves['weight']
            weight_leaf.attrs['AXES'] = ','.join(ordered_axes)
            logging.info("Created soltab {}".format(soltab))

    def delete_soltab(self, soltab):
        if soltab not in self.soltabs:
            raise ValueError("Soltab {} not in solset {}.".format(soltab, self.current_solset))
        with self:
            solset_group = self._H.root._v_groups[self.current_solset]
            soltab_group = solset_group._v_groups[soltab]
            soltab_group._f_remove(recursive=True)

    def delete_solset(self,solset):
        if solset not in self.solsets:
            raise ValueError("Solset {} appears not to exist.".format(solset))
        with self:
            solset_group = self._H.root._v_groups[solset]
            solset_group._f_remove(recursive=True)
        if solset == self.current_solset:
            logging.warn("Setting current solset to None because you deleted it.")
            self._current_solset = None
        logging.info("Deleted solset {}.".format(solset))

    def add_antenna_table(self):
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]
            class Antenna(tb.IsDescription):
                name = tb.StringCol(16, pos=1)
                position = tb.Float64Col(shape=3,dflt=0.0, pos=2)
                #tb.Col(np.float64,3, np.zeros(3, dtype=np.float64),pos=2)

            # descriptor = np.dtype([('name', np.str_, 16), ('position', np.float64, 3)])
            self._H.create_table(solset_group, 'antenna', Antenna,
                                         title='Antenna names and positions', expectedrows=62)
            logging.info("Created antenna table.")

    def add_directions_table(self):
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]

            class Direction(tb.IsDescription):
                name = tb.StringCol(128, pos=1)
                dir = tb.Float64Col(shape=2,dflt=0.0, pos=2)
                #tb.Col(np.float64, 2, np.zeros(2, dtype=np.float64), pos=2)

            # descriptor = np.dtype([('name', np.str_, 16), ('position', np.float64, 3)])
            self._H.create_table(solset_group, 'source', Direction,
                                 title='Direction names and directions', expectedrows=35)
            logging.info("Created direction table.")


    def set_antennas(self, antenna_labels, antennas):
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]
            if 'antenna' not in solset_group._v_leaves:
                self.add_antenna_table()
            antenna_table = solset_group._v_leaves['antenna']
            antenna_table.remove_rows(0)
            antenna_table.append(list(zip(antenna_labels, antennas)))
            logging.info("Set the antenna table.")

    def set_directions(self, patch_names, directions):
        if patch_names is None:
            patch_names = ["patch_{:03d}".format(i) for i in range(len(directions))]
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]
            if 'source' not in solset_group._v_leaves:
                self.add_directions_table()
            direction_table = solset_group._v_leaves['source']
            direction_table.remove_rows(0)
            direction_table.append(list(zip(patch_names, directions)))
            logging.info("Set the direction table.")

    def save_array_file(self, array_file):
        import time
        with self:
            ants = self._solset.getAnt()
            labels = []
            locs = []
            for label, pos in ants.items():
                labels.append(label)
                locs.append(pos)
            Na = len(labels)
        with open(array_file, 'w') as f:
            f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c", time.localtime())))
            f.write('# ITRS(m)\n')
            f.write('# X\tY\tZ\tlabels\n')
            i = 0
            while i < Na:
                f.write(
                    '{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i][0], locs[i][1], locs[i][2], labels[i]))
                if i < Na - 1:
                    f.write('\n')
                i += 1
    @property
    def antennas(self):
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]
            if 'antenna' not in solset_group._v_leaves:
                self.add_antenna_table()
            antenna_table = solset_group._v_leaves['antenna']
            return antenna_table.col('name'), antenna_table.col('position')

    @property
    def directions(self):
        with self:
            if self.current_solset is None:
                raise ValueError("Current solset is None.")
            solset_group = self._H.root._v_groups[self.current_solset]
            if 'source' not in solset_group._v_leaves:
                self.add_directions_table()
            direction_table = solset_group._v_leaves['source']
            return direction_table.col('name'), direction_table.col('dir')

    def __repr__(self):

        def grouper(n, iterable, fillvalue=None):
            "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return itertools.zip_longest(*args, fillvalue=fillvalue)
        _temp_solset = self.current_solset
        info = ""
        for solset in self.solsets:
            self.current_solset = solset
            info += "=== solset: {} ===\n".format(solset)
            info += "Directions: \n"
            for i, src_name1 in enumerate(zip(*self.directions)):
                info += "{} -> {}\t{}\n".format(i, src_name1[0], list(src_name1[1]))

            info += "\nStations: \n"
            for i, ant1 in enumerate(zip(*self.antennas)):
                info += "{} -> {}\t{}\n".format(i, ant1[0],list(ant1[1]))
            for soltab in self.soltabs:
                info += "== soltab: {} ==\n".format(soltab)
                shape = [len(axis_vals) for axis_vals, axis in zip(*self.soltab_axes(soltab))]
                axes = [axis for axis_vals, axis in zip(*self.soltab_axes(soltab))]
                info += "shape: {}\n".format(shape)
                info += "axes: {}\n".format(axes)
                # for axis_vals, axis, size, dtype in zip(*self.soltab_axes(soltab)):
                #     info += "Axis: {} {} {}\n{}\n".format(axis,size, dtype,list(axis_vals))
        self.current_solset = _temp_solset
        return info

    def select(self,**axes):
        self._selection = {}
        for axis_name in self.axes_order:
            if axis_name not in axes.keys():
                continue
            self._selection[axis_name] = axes[axis_name]

    def select_all(self):
        self._selection = None

    @property
    def allowed_soltabs(self):
        return ['phase', 'amplitude', 'tec', 'clock']

    def soltab_axes(self,soltab):
        with self:
            if soltab not in self.soltabs:
                logging.warn('Soltab {} does not exist.'.format(soltab))
                return
            with self:
                if self.current_solset is None:
                    raise ValueError("Current solset is None.")
                solset_group = self._H.root._v_groups[self.current_solset]
                soltab_group = solset_group._v_groups[soltab]
                val_leaf = soltab_group._v_leaves['val']
                try:
                    axes = val_leaf.attrs['AXES'].split(',')
                except TypeError:
                    axes = [s.decode() for s in val_leaf.attrs['AXES'].split(b',')]
                shape = []
                type = []
                vals = []
                for axis in axes:
                    axis_vals = soltab_group._v_leaves[axis].read()
                    vals.append(axis_vals)
                    shape.append(len(axis_vals))
                    type.append(np.array(axis_vals).dtype)
                return vals, axes

    def get_selection(self,soltab):
        if self._selection is None:
            self._selection = {}
        if soltab not in self.soltabs:
            raise ValueError('Soltab {} does not exist.'.format(soltab))
        if self.current_solset is None:
            raise ValueError("Current solset is None.")

        selection = []
        for axis_val, axis in zip(*self.soltab_axes(soltab)):
            axis_selection = self._selection.get(axis, None)
            if axis_selection is None:
                selection.append(slice(None,None,None))
            elif isinstance(axis_selection, slice):
                selection.append(axis_selection)
            elif isinstance(axis_selection, (tuple, list)):
                selection.append(list(axis_selection))
            elif isinstance(axis_selection, str):
                found = list(np.where(np.array([re.search(axis_selection,v.astype(type(axis_selection))) is not None for v in axis_val],dtype=np.bool))[0])
                selection.append(found)
            else:
                selection.append(np.searchsorted(axis_val, axis_selection.astype(axis_val.dtype)))
                # raise ValueError("Selection {} not supported.".format(axis_selection))

        # replace all lists with slices if possible: limitation of only one list per indexing
        def _is_jump_contiguous(sel,n):
            if len(range(int(sel[0]), int(sel[-1])+1, n)) != len(sel):
                return False
            return np.all(np.equal(list(range(int(sel[0]), int(sel[-1])+1, n)), sel))
        corrected_selection = []
        for sel in selection:
            _sel = sel
            if isinstance(sel, list):
                for i in sel:
                    if int(i) != i:
                        raise ValueError("One of the list elements is not an int. {}".format(i))
                for n in range(1,len(sel)+1,1):
                    if _is_jump_contiguous(sel,n):
                        _sel = slice(int(sel[0]), int(sel[-1])+1, n)
                        break
            corrected_selection.append(_sel)

        num_lists = sum([1 if isinstance(sel,list) else 0 for sel in corrected_selection])
        if num_lists > 1:
            raise IndexError("Due to a limitation, only one fancy indexing can be applied per pytables getattr.")
        return tuple(corrected_selection)

    def __getattr__(self, tab):
        """
        Links any attribute with an "axis name" to getValuesAxis("axis name")
        also links val and weight to the relative arrays.
        Parameter
        ----------
        axis : str
            The axis name.
        """
        #        with self:
        #            tabs = self._solset.getSoltabNames()
        tabs = self.allowed_soltabs
        tabs = ["weights_{}".format(t) for t in tabs] + ["axes_{}".format(t) for t in tabs] + tabs
        weight = False
        axes = False
        if tab in tabs:
            if tab.startswith("weights_"):
                tab = "".join(tab.split('weights_')[1:])
                weight = True
            if tab.startswith("axes_"):
                tab = "".join(tab.split('axes_')[1:])
                axes = True
            with self:
                soltab = "{}000".format(tab)
                selection = self.get_selection(soltab)
                if not axes:
                    solset_group = self._H.root._v_groups[self.current_solset]
                    soltab_group = solset_group._v_groups[soltab]
                    if weight:
                        leaf = soltab_group._v_leaves['weight']
                    else:
                        leaf = soltab_group._v_leaves['val']
                    out_axes = {name:np.array(vals)[selection[i]] for i,(vals,name) in enumerate(zip(*self.soltab_axes(soltab)))}
                    out_vals = leaf.__getitem__(selection)
                    return out_vals, out_axes
                else:
                    out_axes = {name:np.array(vals)[selection[i]] for i,(vals,name) in enumerate(zip(*self.soltab_axes(soltab)))}
                    return out_axes
        else:
            return object.__getattribute__(self, tab)

    def __setattr__(self, tab, value):
        """
        Links any attribute with an "axis name" to getValuesAxis("axis name")
        also links val and weight to the relative arrays.
        Parameter
        ----------
        axis : str
            The axis name.
        value : array
            The array of the right shape for selection.
        """

        #        with self:
        #            tabs = self._solset.getSoltabNames()
        tabs = self.allowed_soltabs
        tabs = ["weights_{}".format(t) for t in tabs] + ["axes_{}".format(t) for t in tabs] + tabs
        weight = False
        axes = False
        if tab in tabs:
            if tab.startswith("weights_"):
                tab = "".join(tab.split('weights_')[1:])
                weight = True
            if tab.startswith("axes_"):
                tab = "".join(tab.split('axes_')[1:])
                axes = True
            with self:
                soltab = "{}000".format(tab)
                selection = self.get_selection(soltab)
                solset_group = self._H.root._v_groups[self.current_solset]
                soltab_group = solset_group._v_groups[soltab]
                if not axes:
                    if weight:
                        leaf = soltab_group._v_leaves['weight']
                    else:
                        leaf = soltab_group._v_leaves['val']
                    # print(tab, selection, value.shape, self._selection)

                    leaf.__setitem__(selection, value)
                else:
                    if not isinstance(value, dict):
                        raise("Axes must come in dict of 'name':vals")
                    for i, (k, v) in enumerate(value.items()):
                        axis_vals = soltab_group._v_leaves[k]
                        axis_vals[selection[i]] = v

        else:
            object.__setattr__(self, tab, value)

    @property
    def ref_ant(self):
        with self:
            antenna_labels, antennas = self.antennas
            return antenna_labels[0]

    @property
    def array_center(self):
        with self:
            _, antennas = self.get_antennas(None)
            center = np.mean(antennas.cartesian.xyz, axis=1)
            center = ac.SkyCoord(x=center[0], y=center[1], z=center[2], frame='itrs')
            return center

    def get_antennas(self, ants):
        with self:
            antenna_labels, antennas = self.antennas

            if ants is None:
                ant_idx = slice(None)
            else:
                ants = np.array(ants).astype(antenna_labels.dtype)
                sorter = np.argsort(antenna_labels)
                ant_idx = np.searchsorted(antenna_labels, ants, sorter=sorter)
            antennas = antennas[ant_idx]
            return antenna_labels[ant_idx], ac.SkyCoord(antennas[:, 0] * au.m, antennas[:, 1] * au.m,
                                                        antennas[:, 2] * au.m, frame='itrs')
    @property
    def pointing_center(self):
        with self:
            _, directions = self.get_directions(None)
            ra_mean = np.mean(directions.transform_to('icrs').ra)
            dec_mean = np.mean(directions.transform_to('icrs').dec)
            dir = ac.SkyCoord(ra_mean, dec_mean, frame='icrs')
            return dir

    def get_directions(self, dirs):
        with self:
            patch_names, directions = self.directions
            if dirs is None:
                dir_idx = slice(None)
            else:
                dirs = np.array(dirs).astype(patch_names.dtype)
                sorter = np.argsort(patch_names)
                dir_idx = np.searchsorted(patch_names, dirs, sorter=sorter)
            directions = directions[dir_idx]
            return patch_names[dir_idx], ac.SkyCoord(directions[:, 0] * au.rad, directions[:, 1] * au.rad, frame='icrs')


    def get_times(self, times):
        """
        times are stored as mjs
        """
        times = at.Time(times / 86400., format='mjd')
        return times.isot, times

    def get_freqs(self, freqs):
        labs = ['{:.1f}MHz'.format(f / 1e6) for f in freqs]
        return np.array(labs), freqs

    def get_pols(self, pols):
        with self:
            return pols, np.arange(len(pols), dtype=np.int32)





