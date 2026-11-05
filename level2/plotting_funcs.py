#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:33:45 2024

@author: robfren
"""

import os
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import numpy as np

from scipy import sparse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nilearn import datasets, surface, plotting
from nilearn.surface import SurfaceImage

def adj_cmap(cmap, exp=1.5):
    # adjust colormap to increase contrast close to 0 (log)
    adj_cmap = cmap(np.arange(cmap.N))

    # pull halves- put in rgb space
    upper_half = adj_cmap[:128,:]
    lower_half = adj_cmap[-1:128:-1,:] # reverse
    # adjust colormap values by exponential scaling
    upper_half = upper_half**exp
    lower_half = lower_half**exp
    # splice arrays back together
    adj_cmap = np.append(upper_half, lower_half[::-1], axis=0)
    adj_cmap = ListedColormap(adj_cmap)

    return adj_cmap

def CurlyBrace_v(ax, ll_corner=(0, 0), width=1, height=1):
    Path = mpath.Path
    verts = np.array(
        [(0, 0), (-.5, 0), (-.5, .2), (-.5, .3), (-.5, .5), (-1, .5), (-.5, .5), (-.5, .7), (-.5, .8), (-.5, 1),
         (0, 1)])
    verts[:, 0] *= width
    verts[:, 1] *= height
    verts[:, 0] += ll_corner[0]
    verts[:, 1] += ll_corner[1]

    cb_patch = mpatches.PathPatch(
        Path(verts,
             [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3,
              Path.LINETO, Path.CURVE3, Path.CURVE3]),
        fc='none', ec='black', clip_on=False, transform=ax.transData)
    return cb_patch


def CurlyBrace_h(ax, ll_corner=(0, 0), width=1, height=1):
    Path = mpath.Path
    verts = np.array(
        [(0, 0), (0, -.5), (.2, -.5), (.3, -.5), (.5, -.5), (.5, -1), (.5, -.5), (.7, -.5), (.8, -.5), (1, -.5),
         (1, 0)])
    verts[:, 0] *= height
    verts[:, 1] *= width
    verts[:, 0] += ll_corner[0]
    verts[:, 1] += ll_corner[1]

    cb_patch = mpatches.PathPatch(
        Path(verts,
             [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3,
              Path.LINETO, Path.CURVE3, Path.CURVE3]),
        fc='none', ec='black', clip_on=False, transform=ax.transData)
    return cb_patch

def FC_heatmap_nodiv(data, vmin, vmax, 
               ax=None, cbar_kw=None, cbarlabel="",
               **kwargs):
    """
    Create FC heatmap.

    Parameters
    ----------
    data
        A 2D numpy array of FC.
    vmin
        minimum value for FC heatmap color mapping
    vmax
        maximum value for FC heatmap color mapping
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    return im, cbar


def FC_heatmap(data, vmin=None, vmax=None, divnorm=True,
               ax=None, cbar_kw={}, cbarlabel="", center=0,
               **kwargs):
    """
    Create FC heatmap.

    Parameters
    ----------
    data
        A 2D numpy array of FC.
    vmin
        minimum value for FC heatmap color mapping
    vmax
        maximum value for FC heatmap color mapping
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    """
    if type(divnorm) != bool:
        raise AttributeError('`divnorm must be True or False`')
        
    if ax is None:
        ax = plt.gca()
    
    if vmin is None:
        vmin=np.nanmin(data)
        
    if vmax is None:
        vmax=np.nanmax(data)
    
    # set cmap to diverging norm
    if divnorm:
        dv = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        
        im = ax.imshow(data, norm=dv, **kwargs)
        
    else:
        im = ax.imshow(data, **{**{'vmin': vmin, 'vmax': vmax}, **kwargs})

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    
    return im, cbar


def map_mat_braced(mat, title, rot, bands, names,  
                   **kwargs):
    fig, ax = plt.subplots(figsize=(11, 10))

    im, cbar = FC_heatmap(
        mat, ax=ax, 
        **kwargs)

    for y, h, bn in zip(bands, np.diff(bands), names):
        cb = CurlyBrace_v(ax, [-1, y + h * 0.025], 8, h * 0.95)
        ax.add_patch(cb)
        if rot:
            ax.text(-10, y + h / 2 - 2, bn, va='top', ha='right', fontsize=10, rotation=45)
            ax.text(y + h / 2 - 2, -10, bn, va='bottom', ha='left', fontsize=10, rotation=45)
        else:
            ax.text(-10, y + h / 2, bn, va='center', ha='right', fontsize=10)
            ax.text(y + h / 2 - 2, -10, bn, va='bottom', fontsize=10, rotation=60)

        cb = CurlyBrace_h(ax, [y + h * 0.025, -1], 8, h * .95)
        ax.add_patch(cb)

    ax.hlines(bands[:-1] - 0.5, *(0, 200), alpha=0.5, lw=0.5, color='k')
    ax.vlines(bands[:-1] - 0.5, *(0, 200), alpha=0.5, lw=0.5, color='k')
    fig.suptitle(f'{title}')

    fig.tight_layout()
    
    return fig, ax, cbar



def get_inward_border_faces(parcel_labels, faces, parcel_id, n_rings=2):
    """
    Find faces forming an inward border band for one parcel.

    Parameters
    ----------
    parcel_labels : (n_vertices,) ndarray
        Integer atlas label at each mesh vertex.

    faces : (n_faces, 3) ndarray
        Mesh triangle indices.

    parcel_id : int
        Atlas label for the parcel of interest.

    n_rings : int
        Number of mesh-neighbor rings to extend inward from the parcel
        boundary. n_rings=0 gives approximately one triangle-wide boundary.

    Returns
    -------
    border_faces : (n_border_faces,) ndarray
        Indices of mesh faces to color.
    """

    n_vertices = len(parcel_labels)

    # ------------------------------------------------------------
    # Build vertex adjacency from triangular mesh
    # ------------------------------------------------------------
    i = faces[:, [0, 1, 2]]
    j = faces[:, [1, 2, 0]]

    rows = np.concatenate([i.ravel(), j.ravel()])
    cols = np.concatenate([j.ravel(), i.ravel()])

    adjacency = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=bool),
            (rows, cols)
        ),
        shape=(n_vertices, n_vertices)
    )

    # ------------------------------------------------------------
    # Vertices belonging to this parcel
    # ------------------------------------------------------------
    parcel_mask = parcel_labels == parcel_id
    parcel_vertices = np.flatnonzero(parcel_mask)

    if len(parcel_vertices) == 0:
        return np.array([], dtype=int)

    # ------------------------------------------------------------
    # Find parcel boundary vertices
    #
    # A parcel vertex is on the boundary if it has at least one
    # mesh neighbor belonging to another parcel.
    # ------------------------------------------------------------
    neighbor_counts_outside = np.asarray(
        adjacency[parcel_vertices][:, ~parcel_mask].sum(axis=1)
    ).ravel()

    boundary_vertices = parcel_vertices[neighbor_counts_outside > 0]

    # ------------------------------------------------------------
    # Walk inward N mesh-neighbor rings
    # ------------------------------------------------------------
    border_vertices = set(boundary_vertices.tolist())
    frontier = boundary_vertices

    for _ in range(n_rings):
        if len(frontier) == 0:
            break

        neighbors = adjacency[frontier].indices

        # Keep ONLY neighbors inside this parcel
        neighbors = neighbors[parcel_mask[neighbors]]

        # Only retain vertices not already included
        neighbors = np.setdiff1d(
            neighbors,
            np.fromiter(border_vertices, dtype=int),
            assume_unique=False
        )

        border_vertices.update(neighbors.tolist())
        frontier = neighbors

    border_vertices = np.fromiter(border_vertices, dtype=int)

    # ------------------------------------------------------------
    # Convert border vertices -> faces
    #
    # First require every vertex of the face to be inside the
    # parcel. This prevents the overlay from bleeding outward.
    #
    # Then require at least one vertex of the face to belong to
    # the inward border band.
    # ------------------------------------------------------------
    face_inside_parcel = parcel_mask[faces].all(axis=1)
    face_touches_border = np.isin(faces, border_vertices).any(axis=1)

    border_faces = np.flatnonzero(
        face_inside_parcel & face_touches_border
    )

    return border_faces

def plot_surf(
    axis_avg,
    contour_bin,
    model_name,
    contour_col='g',
    nilearn_kwargs=None,
    vmin=None,
    vmax=None,
    sym_cbar=False,
    mesh_qual='high',
    bval=None,
    border_rings=6,
    border_debug=False,
):


    if nilearn_kwargs is None:
        nilearn_kwargs = {}
    else:
        nilearn_kwargs = nilearn_kwargs.copy()


    # ------------------------------------------------------------
    # Split parcel-level values
    # ------------------------------------------------------------

    lh_v = axis_avg[:100]
    rh_v = axis_avg[100:]

    contour_bin_l = contour_bin[:100]
    contour_bin_r = contour_bin[100:]


    # ------------------------------------------------------------
    # Schaefer directory
    # ------------------------------------------------------------

    bdir = (
        'POINT/TO/LOCAL/CLONE/OF/SCHAEFER_PARC'

        # e.g.
        # 'CBIG-0.27.2-update_reorder_func/'
        # 'stable_projects/brain_parcellation/'
        # 'Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3'
    )


    # ------------------------------------------------------------
    # Mesh quality
    # ------------------------------------------------------------

    if mesh_qual.lower() == 'high':
        fs = 'fsaverage'

    elif mesh_qual.lower() == 'med':
        fs = 'fsaverage6'

    elif mesh_qual.lower() == 'low':
        fs = 'fsaverage5'

    else:
        raise ValueError(
            '`mesh_qual` must be "high", "med", or "low"'
        )


    # ------------------------------------------------------------
    # Load atlas labels
    # ------------------------------------------------------------

    schaefer_l_lab = surface.load_surf_data(
        f'{bdir}/{fs}/label/'
        'lh.Schaefer2018_200Parcels_17Networks_order.annot'
    )

    schaefer_r_lab = surface.load_surf_data(
        f'{bdir}/{fs}/label/'
        'rh.Schaefer2018_200Parcels_17Networks_order.annot'
    )


    # ------------------------------------------------------------
    # Load fsaverage meshes
    # ------------------------------------------------------------

    fsaverage = datasets.fetch_surf_fsaverage(fs)


    # ------------------------------------------------------------
    # Convert parcel values to vertex values
    # ------------------------------------------------------------

    def parcel_to_vertex(parcel_values, labels):

        mapping = np.zeros(
            101,
            dtype=float
        )

        mapping[1:101] = parcel_values

        return mapping[labels]


    left_betas = parcel_to_vertex(
        lh_v,
        schaefer_l_lab
    )

    right_betas = parcel_to_vertex(
        rh_v,
        schaefer_r_lab
    )


    # ------------------------------------------------------------
    # Color limits
    # ------------------------------------------------------------

    if vmin is None:
        vmin = np.nanmin(axis_avg)

    if vmax is None:
        vmax = np.nanmax(axis_avg)


    if sym_cbar:

        if bval is None:
            bval = max(
                abs(vmin),
                abs(vmax)
            )

        vmin = -bval
        vmax = bval

    else:

        if vmin < 0:
            vmin = 0


    # ------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------

    cmap = nilearn_kwargs.get(
        'cmap',
        plt.cm.RdBu_r
    )

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)


    threshold = nilearn_kwargs.get(
        'threshold',
        None
    )


    norm = mcolors.Normalize(
        vmin=vmin,
        vmax=vmax
    )


    border_rgba = np.array(
        mcolors.to_rgba(
            contour_col
        )
    )


    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------

    outd = (
        f'figs/surface_plots/'
        f'{model_name}'
    )

    os.makedirs(
        outd,
        exist_ok=True
    )


    # ------------------------------------------------------------
    # Selected parcel IDs
    # ------------------------------------------------------------

    reg = {
        'left':
            np.flatnonzero(
                contour_bin_l == 1
            ) + 1,

        'right':
            np.flatnonzero(
                contour_bin_r == 1
            ) + 1,
    }


    # ------------------------------------------------------------
    # Hemisphere loop
    # ------------------------------------------------------------

    for hemi, betas, parcel_labels in zip(

        ['left', 'right'],

        [
            left_betas,
            right_betas
        ],

        [
            schaefer_l_lab,
            schaefer_r_lab
        ]
    ):

        # --------------------------------------------------------
        # Mesh
        # --------------------------------------------------------

        coords, faces = surface.load_surf_mesh(
            fsaverage[
                f'infl_{hemi}'
            ]
        )


        # --------------------------------------------------------
        # Statistical value per FACE
        #
        # Mean of the three vertices.
        # This is roughly analogous to Nilearn's smooth averaging.
        # --------------------------------------------------------

        face_values = np.nanmean(
            betas[faces],
            axis=1
        )


        # --------------------------------------------------------
        # Convert statistics -> RGBA colors
        # --------------------------------------------------------

        facecolors = cmap(
            norm(face_values)
        )


        # --------------------------------------------------------
        # Threshold
        #
        # Instead of transparency, make subthreshold faces a
        # neutral light gray. This avoids another transparency/
        # depth-rendering complication.
        # --------------------------------------------------------

        if threshold is not None:

            thresh_mask = (
                np.abs(face_values)
                <
                threshold
            )

            facecolors[
                thresh_mask
            ] = np.array([
                0.85,
                0.85,
                0.85,
                1.0
            ])


        # --------------------------------------------------------
        # Build inward-border face mask
        # --------------------------------------------------------

        all_border_faces = np.zeros(
            len(faces),
            dtype=bool
        )


        for region_id in reg[hemi]:

            region_id = int(
                region_id
            )


            face_idx = get_inward_border_faces(
                parcel_labels=parcel_labels,
                faces=faces,
                parcel_id=region_id,
                n_rings=border_rings,
            )


            if border_debug:

                print(
                    f'{hemi} | '
                    f'parcel {region_id} | '
                    f'rings={border_rings} | '
                    f'{len(face_idx)} faces'
                )


            all_border_faces[
                face_idx
            ] = True


        if border_debug:

            print(
                f'{hemi}: '
                f'{all_border_faces.sum()} '
                f'total border faces'
            )


        # --------------------------------------------------------
        # IMPORTANT:
        # overwrite the statistical face colors BEFORE plotting.
        #
        # This means border and statistical map are ONE surface.
        # --------------------------------------------------------

        facecolors[
            all_border_faces
        ] = border_rgba


        # --------------------------------------------------------
        # Views
        # --------------------------------------------------------

        for view in [
            'medial',
            'lateral'
        ]:

            fig = plt.figure(
                figsize=(10, 10)
            )

            ax = fig.add_subplot(
                111,
                projection='3d'
            )


            # ----------------------------------------------------
            # SINGLE cortical surface
            # ----------------------------------------------------

            surf = ax.plot_trisurf(

                coords[:, 0],
                coords[:, 1],
                coords[:, 2],

                triangles=faces,

                linewidth=0,
                antialiased=False,
                shade=False,
            )


            surf.set_facecolors(
                facecolors
            )


            # ----------------------------------------------------
            # Camera orientation
            # ----------------------------------------------------

            if hemi == 'left':

                if view == 'lateral':
                    ax.view_init(
                        elev=0,
                        azim=180
                    )

                else:
                    ax.view_init(
                        elev=0,
                        azim=0
                    )

            else:

                if view == 'lateral':
                    ax.view_init(
                        elev=0,
                        azim=0
                    )

                else:
                    ax.view_init(
                        elev=0,
                        azim=180
                    )


            # ----------------------------------------------------
            # Equal-ish aspect
            # ----------------------------------------------------

            xmid = (
                coords[:, 0].max()
                +
                coords[:, 0].min()
            ) / 2

            ymid = (
                coords[:, 1].max()
                +
                coords[:, 1].min()
            ) / 2

            zmid = (
                coords[:, 2].max()
                +
                coords[:, 2].min()
            ) / 2


            radius = max(
                np.ptp(coords[:, 0]),
                np.ptp(coords[:, 1]),
                np.ptp(coords[:, 2]),
            ) / 2


            ax.set_xlim(
                xmid - radius,
                xmid + radius
            )

            ax.set_ylim(
                ymid - radius,
                ymid + radius
            )

            ax.set_zlim(
                zmid - radius,
                zmid + radius
            )


            ax.set_box_aspect(
                (
                    1,
                    1,
                    1
                )
            )


            ax.set_axis_off()


            # ----------------------------------------------------
            # Colorbar
            # ----------------------------------------------------

            scalar_map = cm.ScalarMappable(
                norm=norm,
                cmap=cmap
            )

            scalar_map.set_array([])


            fig.colorbar(
                scalar_map,
                ax=ax,
                shrink=0.55,
                pad=0.02
            )


            # ----------------------------------------------------
            # Save
            # ----------------------------------------------------

            output_file = (
                f'{outd}/'
                f'{model_name}_'
                f'{view}_'
                f'{hemi}_'
                f'contour_{mesh_qual}Q.png'
            )


            fig.savefig(
                output_file,
                dpi=300,
                bbox_inches='tight',
                transparent=False
            )


            plt.close(
                fig
            )

def plot_surf_orig(axis_avg, contour_bin, model_name, contour_col = 'g', nilearn_kwargs = {}, vmin=None, vmax=None, sym_cbar=False, mesh_qual='high', bval=None):

    lh_v = axis_avg[:100]
    rh_v = axis_avg[100:]
    contour_bin_l = contour_bin[:100]
    contour_bin_r = contour_bin[100:]

    # this needs to point to local copy of freesurfer schaefer parcellation - get all 3 for different mesh qualities.
    # found here: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3
 
    bdir = '/local/path/to/schaefer/freesurfer/parcellation/FreeSurfer5.3'
    
    if mesh_qual.lower() == 'high':
        fs = 'fsaverage'
    elif mesh_qual.lower() == 'med':
        fs = 'fsaverage6'
    elif mesh_qual.lower() == 'low':
        fs = 'fsaverage5'
    else:
        raise ValueError('`mesh_qual` must be "high", "med", or "low"')

    schaefer_l_lab = surface.load_surf_data(f'{bdir}/{fs}/label/lh.Schaefer2018_200Parcels_17Networks_order.annot')
    schaefer_r_lab = surface.load_surf_data(f'{bdir}/{fs}/label/rh.Schaefer2018_200Parcels_17Networks_order.annot')

    # Retrieve fsaverage5 surface dataset for the plotting background. 
    fsaverage = datasets.fetch_surf_fsaverage(fs)
    fs_mapping = datasets.load_fsaverage(fs)

    # alter labels to reflect the collapsed regions
    mapping_l = {k: v for k, v in zip(range(1,101), lh_v)}
    mapping_r = {k: v for k, v in zip(range(1,101), rh_v)}

    # left
    k = np.array(list(mapping_l.keys()))
    v = np.array(list(mapping_l.values()))

    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype)
    mapping_ar[k] = v
    left_betas = mapping_ar[schaefer_l_lab]

    # right
    k = np.array(list(mapping_r.keys()))
    v = np.array(list(mapping_r.values()))

    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype)
    mapping_ar[k] = v
    right_betas = mapping_ar[schaefer_r_lab]
    
    if vmin is None:
        vmin = np.min(axis_avg)
    if vmax is None:
        vmax = np.max(axis_avg)
    if sym_cbar:
        if bval is None:
            bval = np.sort((np.abs(vmin), np.abs(vmax)))[1]
        vmax = bval
        vmin = -bval
    else:
        if vmin < 0:
            vmin = 0
        
    
    outd = f'figs/surface_plots/{model_name}'    
    if not os.path.exists(outd):
        os.makedirs(outd)

    # force threshold low to show full results, contour maps threshold
    nilearn_kwargs['threshold']=0.001

    reg = {}

    reg['left'] = np.argwhere(contour_bin_l == 1)+1
    reg['right'] = np.argwhere(contour_bin_r == 1)+1

    sch_atlas = SurfaceImage(mesh=fs_mapping['inflated'],
                                data={"left": schaefer_l_lab,
                                    "right": schaefer_r_lab})

    col = contour_col

    for hemi, betas in zip(['left', 'right'],[left_betas, right_betas]):
        regions_indices = reg[hemi]
        cols = [col for i in regions_indices]

        for view in ['medial', 'lateral']:
            # generate new figure- larger than default for saving
            fig, axes = plt.subplots(figsize=(10,10), subplot_kw={'projection': '3d'})
            plotting.plot_surf_stat_map(
                        fsaverage[f'infl_{hemi}'], betas, hemi=hemi,
                        colorbar=True, symmetric_cbar=sym_cbar, 
                        darkness=None, view=view, vmax=vmax, vmin=vmin,
                        figure=fig, axes=axes,
                        **nilearn_kwargs)
            plotting.plot_surf_contours(
                        roi_map=sch_atlas,
                        hemi=hemi,
                        view=view,
                        levels=regions_indices,
                        figure=fig, axes=axes,
                        colors=cols,
                        output_file=f'{outd}/{model_name}_{view}_{hemi}_contour_{mesh_qual}Q.png',
                    )
            
