# -*- coding: utf-8 -*-
#!/bin/env python3

# Copyright (C) 2018 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>
# URL: https://github.com/gabylaunay/pyDSA_core

# This file is part of pyDSA_core

# pyDSA_core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import numpy as np
import json
import os
import warnings
from IMTreatment import TemporalScalarFields
from IMTreatment.utils import ProgressCounter
from IMTreatment.stabilize import Stabilizer
import IMTreatment.plotlib as pplt
from . import image
from . import baseline
from . import temporaldropedges as tde
from . import dropedges


"""  """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__credits__ = ""
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"


class TemporalImages(TemporalScalarFields):
    def __init__(self, filepath=None, cache_infos=True):
        super().__init__()
        self.baseline = None
        self.field_type = image.Image
        if filepath is not None:
            self.filepath = os.path.abspath(filepath)
            if os.path.isdir(self.filepath):
                self.infofile_path = os.path.join(self.filepath, "info.info")
            else:
                self.infofile_path = os.path.splitext(self.filepath)[0] + ".info"
            self.cache_infos = cache_infos
        else:
            self.filepath = None
            self.infofile_path = None
            self.cache_infos = False

    def add_field(self, field, time=0, unit_times="", copy=True):
        super().add_field(field=field, time=time, unit_times=unit_times,
                          copy=copy)
        if self.baseline:
            self.fields[-1].baseline = self.baseline
        else:
            self.baseline = self.fields[-1].baseline

    def set_baseline(self, pt1, pt2):
        """
        Set the drop baseline.
        """
        for i in range(len(self.fields)):
            self.fields[i].set_baseline(pt1=pt1, pt2=pt2)
        self.baseline = baseline.Baseline(
            [pt1, pt2],
            xmax=self.axe_x[-1],
            xmin=self.axe_x[0])
        if self.cache_infos:
            self._dump_infos()

    def set_evolving_baseline(self, baseline1, baseline2):
        """
        Set a linearly evolving baseline.
        """
        base1_pt1 = np.asarray(baseline1[0])
        base1_pt2 = np.asarray(baseline1[1])
        base2_pt1 = np.asarray(baseline2[0])
        base2_pt2 = np.asarray(baseline2[1])
        imax = len(self.fields) - 1
        for i in range(len(self.fields)):
            ratio = i/imax
            pt1 = base1_pt1*(1 - ratio) + base2_pt1*ratio
            pt2 = base1_pt2*(1 - ratio) + base2_pt2*ratio
            self.fields[i].set_baseline(pt1, pt2)
        self.baseline = "evolving"
        if self.cache_infos:
            self._dump_infos()

    def set_origin(self, x, y):
        """
        Modify the axis in order to place the origin at the given point (x, y)

        Parameters
        ----------
        x : number
        y : number
        """
        for im in self.fields:
            im.set_origin(x, y)
        self.axe_x = self.fields[0].axe_x
        self.axe_y = self.fields[0].axe_y
        self.baseline = self.fields[0].baseline
        if self.cache_infos:
            self._dump_infos()

    def choose_baseline(self, ind_image=None):
        """
        Choose baseline position interactively.

        Select as many points as you want (click on points to delete them),
        and close the figure when done.
        """
        #
        if ind_image is None:
            ind_image = int(len(self.fields)/2)
        self.baseline = self.fields[ind_image].choose_baseline()
        # No baseline set
        if self.baseline is None:
            return None
        for i in range(len(self.fields)):
            self.fields[i].baseline = self.baseline
        if self.cache_infos:
            self._dump_infos()
        return self.baseline

    def track_baseline(self, ind_image=0, orb_kwargs={}, mode='continuous',
                       verbose=False, dense=False):
        """
        Track the baseline from the wanted frame.

        Use opencv features tracking algorithm.
        """
        stab = Stabilizer(self, orb_kwargs=orb_kwargs, mode=mode)
        stab._compute_transform(dense=dense, opt_flow_args={}, verbose=verbose)
        stab._smooth_transform(smooth_size=0, verbose=verbose)
        pt1 = self[0].baseline.pt1
        pt2 = self[0].baseline.pt2
        pts = stab._apply_transform_to_point([pt1, pt2], from_frame=ind_image,
                                             verbose=verbose)
        # apply new baselines
        for i, im in enumerate(self.fields):
            im.set_baseline(pt1=pts[0][i], pt2=pts[1][i])
        # set baseline to evolving
        self.baseline = "evolving"
        # return
        return stab

    def display(self, *args, **kwargs):
        super().display(*args, **kwargs)
        if isinstance(self.baseline, baseline.Baseline):
            self.baseline.display()
        elif self.baseline == "evolving":
            # Display evolving baseline
            xs = []
            ys = []
            for field in self.fields:
                bl = field.baseline
                xs.append([bl.pt1[0], bl.pt2[0]])
                ys.append([bl.pt1[1], bl.pt2[1]])
            db = pplt.Displayer(xs, ys, color=self[0].colors[0])
            pplt.ButtonManager(db)
        else:
            pass

    def scale_interactive(self, indice=None):
        """
        Scale the Images interactively.

        Parameters
        ==========
        indice: integer
            Frame number on which do the scaling.
            (default to the mid one).
        """
        indice = indice or int(len(self.fields)/2)
        scale, unit = self.fields[indice].scale_interactive()
        if scale is None and unit is None:
            return None
        self.scale(scalex=scale, scaley=scale, inplace=True)
        self.unit_x = unit
        self.unit_y = unit
        if self.cache_infos:
            self._dump_infos()

    def scale(self, scalex=None, scaley=None, scalev=None, scalet=None,
              inplace=False):
        """
        Scale the Fields.

        Parameters
        ----------
        scalex, scaley, scalev : numbers or Unum objects
            Scale for the axis and the values.
        inplace : boolean
            .
        """
        if inplace:
            tmp_f = self
        else:
            tmp_f = self.copy()
        old_dx = self.dx
        old_dy = self.dy
        # scale the scalarfields
        super(TemporalImages, tmp_f).scale(scalex=scalex, scaley=scaley,
                                           scalev=scalev, scalet=scalet,
                                           inplace=True)
        new_dx = self.dx
        new_dy = self.dy
        if self.baseline is not None:
            tmp_f.baseline.scale(scalex=new_dx/old_dx,
                                 scaley=new_dy/old_dy)
            for f in self.fields:
                f.baseline = tmp_f.baseline
        return tmp_f

    def _dump_infos(self):
        # Gather old information if necessary
        if os.path.isfile(self.infofile_path):
            with open(self.infofile_path, 'r') as f:
                dic = json.load(f)
        else:
            dic = {}
        # Update with new values
        unit_x = self.unit_x.strUnit()[1:-1]
        unit_y = self.unit_y.strUnit()[1:-1]
        if self.baseline != "evolving" and self.baseline is not None:
            pt1 = list(self.baseline.pt1)
            pt2 = list(self.baseline.pt2)
        else:
            pt1 = None
            pt2 = None
        new_dic = {"dx": self.dx,
                   "dy": self.dy,
                   "x0": self.axe_x[0],
                   "y0": self.axe_y[0],
                   "baseline_pt1": pt1,
                   "baseline_pt2": pt2,
                   "unit_x": unit_x,
                   "unit_y": unit_y}
        dic.update(new_dic)
        # Write back infos
        with open(self.infofile_path, 'w+') as f:
            json.dump(dic, f)

    def _import_infos(self):
        if not os.path.isfile(self.infofile_path):
            return None
        try:
            with open(self.infofile_path, 'r') as f:
                dic = json.load(f)
        except:
            warnings.warn('Corrupted infofile, reinitializing...')
            os.remove(self.infofile_path)
            return None
        # Update with infos
        dx = dic['dx']
        dy = dic['dy']
        self.scale(scalex=dx/(self.axe_x[1] - self.axe_x[0]),
                   scaley=dy/(self.axe_y[1] - self.axe_y[0]),
                   inplace=True)
        if self.unit_x.strUnit()[1:-1] != dic['unit_x']:
            self.unit_x = dic['unit_x']
        if self.unit_y.strUnit()[1:-1] != dic['unit_y']:
            self.unit_y = dic['unit_y']
        try:  # Compatibility
            x0 = dic['x0']
            y0 = dic['y0']
            self.axe_x += (x0 - self.axe_x[0])
            self.axe_y += (y0 - self.axe_y[0])
        except KeyError:
            pass
        base1 = dic['baseline_pt1']
        base2 = dic['baseline_pt2']
        if base1 is not None and base2 is not None:
            self.set_baseline(base1, base2)

    def edge_detection_canny(self, threshold1=None, threshold2=None,
                             base_max_dist=30, size_ratio=.5,
                             ignored_pixels=2, nmb_edges=2,
                             keep_exterior=True,
                             dilatation_steps=1,
                             smooth_size=None,
                             iteration_hook=None,
                             verbose=False):
        """
        Perform edge detection.

        Parameters
        ==========
        threshold1, threshold2: integers
            Thresholds for the Canny edge detection method.
            (By default, inferred from the data histogram)
        base_max_dist: integers
            Maximal distance (in pixel) between the baseline and
            the beginning of the drop edge (default to 30).
        nmb_edges: integer
            Number of maximum expected edges (default to 2).
        size_ratio: number
            Minimum size of edges, regarding the bigger detected one
            (default to 0.5).
        keep_exterior: boolean
            If True (default), only keep the exterior edges.
        smooth_size: number
            If specified, the image is smoothed before
            performing the edge detection.
            (can be useful to put this to 1 to get rid of compression
             artefacts on images).
        dilatation_steps: positive integer
            Number of dilatation/erosion steps.
            Increase this if the drop edges are discontinuous.
        iteration_hook: function
            Hook run at each iterations with the iteration number
            and the total number of iterations planned.
        """
        # check
        if self.baseline is None:
            raise Exception('You have to define a baseline first.')
        #
        all_edge_empty = True
        pts = tde.TemporalDropEdges()
        pts.baseline = self.baseline
        if verbose:
            pg = ProgressCounter(init_mess="Detecting drop edges",
                                 nmb_max=len(self.fields),
                                 name_things='images',
                                 perc_interv=5)
        for i in range(len(self.fields)):
            try:
                pt = self.fields[i].edge_detection_canny(
                    threshold1=threshold1,
                    threshold2=threshold2,
                    base_max_dist=base_max_dist,
                    size_ratio=size_ratio,
                    ignored_pixels=ignored_pixels,
                    keep_exterior=keep_exterior,
                    nmb_edges=nmb_edges,
                    dilatation_steps=dilatation_steps,
                    smooth_size=smooth_size)
                all_edge_empty = False
            except Exception:
                pt = dropedges.DropEdges(xy=[], im=self, type='canny')
            pts.add_pts(pt, time=self.times[i], unit_times=self.unit_times)
            if iteration_hook is not None:
                iteration_hook(i, len(self.fields))
            if verbose:
                pg.print_progress()
        # check if edges has been detected
        if all_edge_empty:
            raise Exception('No edges could be detected. You should'
                            ' check the baseline position.')
        return pts

    def edge_detection_contour(self, size_ratio=.5, nmb_edges=2,
                               level=0.5, ignored_pixels=2,
                               iteration_hook=None,
                               verbose=False):
        """
        Perform edge detection.

        Parameters
        ==========
        nmb_edges: integer
            Number of maximum expected edges (default to 2).
        level: number
            Normalized level of the drop contour.
            Should be between 0 (black) and 1 (white).
            Default to 0.5.
        size_ratio: number
            Minimum size of edges, regarding the bigger detected one
            (default to 0.5).
        ignored_pixels: integer
            Number of pixels ignored around the baseline
            (default to 2).
            Putting a small value to this allow to avoid
            small surface defects to be taken into account.
        """
        pts = tde.TemporalDropEdges()
        pts.baseline = self.baseline
        if verbose:
            pg = ProgressCounter("Detecting drop edges", "Done",
                                 len(self.fields), 'images', 5)
        for i in range(len(self.fields)):
            try:
                pt = self.fields[i].edge_detection_contour(nmb_edges=nmb_edges,
                                                           level=level,
                                                           ignored_pixels=ignored_pixels,
                                                           size_ratio=size_ratio)
            except Exception:
                pt = dropedges.DropEdges(xy=[], im=self, type='contour')
            pts.add_pts(pt, time=self.times[i], unit_times=self.unit_times)
            if verbose:
                pg.print_progress()
            if iteration_hook is not None:
                iteration_hook(i, len(self.fields))
        return pts

    edge_detection = edge_detection_canny
