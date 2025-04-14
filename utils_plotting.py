# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:42:28 2024

@author: dowel

This script contains utility functions for plotting

List and description here:
    coloured_line(x,y,c,ax,**lc_kwargs) : plots a line with varying colour segments specified
    coloured_line_between_pts(x, y, c, ax, **lc_kwargs):
        
    
Add new functions as and when


"""
#%%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection

#%%
class uplt:
    def _init__(self):
        self.version = 241009
        
    def plot_arrows(x,y,phase,length, **lc_kwargs):
        dx = np.sin(phase)*length
        dy = np.cos(phase)*length
        for i,tx in enumerate(x):
            plt.plot([tx,tx+dx[i]],[y[i],y[i]+dy[i]],**lc_kwargs)
    def coloured_line(x, y, c, ax, **lc_kwargs):
        """
        Taken from here: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
        
        
        Plot a line with a color specified along the line by a third value.
    
        It does this by creating a collection of line segments. Each line segment is
        made up of two straight lines each connecting the current (x, y) point to the
        midpoints of the lines connecting the current point with its two neighbors.
        This creates a smooth line with no gaps between the line segments.
    
        Parameters
        ----------
        x, y : array-like
            The horizontal and vertical coordinates of the data points.
        c : array-like
            The color values, which should be the same size as x and y.
        ax : Axes
            Axis object on which to plot the colored line.
        **lc_kwargs
            Any additional arguments to pass to matplotlib.collections.LineCollection
            constructor. This should not include the array keyword argument because
            that is set to the color argument. If provided, it will be overridden.
    
        Returns
        -------
        matplotlib.collections.LineCollection
            The generated line collection representing the colored line.
        """
        if "array" in lc_kwargs:
            warnings.warn('The provided "array" keyword argument will be overridden')
    
        # Default the capstyle to butt so that the line segments smoothly line up
        default_kwargs = {"capstyle": "butt"}
        default_kwargs.update(lc_kwargs)
    
        # Compute the midpoints of the line segments. Include the first and last points
        # twice so we don't need any special syntax later to handle them.
        x = np.asarray(x)
        y = np.asarray(y)
        x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
        y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
    
        # Determine the start, middle, and end coordinate pair of each line segment.
        # Use the reshape to add an extra dimension so each pair of points is in its
        # own list. Then concatenate them to create:
        # [
        #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
        #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
        #   ...
        # ]
        coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
        coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
        coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
        segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    
        lc = LineCollection(segments, **default_kwargs)
        lc.set_array(c)  # set the colors of each segment
    
        return ax.add_collection(lc)
    def coloured_line_simple(x,y,c,cmap,cmin,cmax):
        import matplotlib as mpl
        from matplotlib import cm
        c_map = plt.get_cmap(cmap)
        cnorm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
        scalarMap = cm.ScalarMappable(cnorm, c_map)
        c_map_rgb = scalarMap.to_rgba(c)
 
        for i in range(len(c)-1):
            x1 = x[i:i+2]
            y1 = y[i:i+2]
            #ca = np.mean(ca[i:i+2])
            plt.plot(x1,y1,color=c_map_rgb[i,:])
            
        
    def coloured_line_between_pts(x, y, c, ax, **lc_kwargs):
        """
        Taken from here:https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
        
        Plot a line with a color specified between (x, y) points by a third value.
    
        It does this by creating a collection of line segments between each pair of
        neighboring points. The color of each segment is determined by the
        made up of two straight lines each connecting the current (x, y) point to the
        midpoints of the lines connecting the current point with its two neighbors.
        This creates a smooth line with no gaps between the line segments.
    
        Parameters
        ----------
        x, y : array-like
            The horizontal and vertical coordinates of the data points.
        c : array-like
            The color values, which should have a size one less than that of x and y.
        ax : Axes
            Axis object on which to plot the colored line.
        **lc_kwargs
            Any additional arguments to pass to matplotlib.collections.LineCollection
            constructor. This should not include the array keyword argument because
            that is set to the color argument. If provided, it will be overridden.
    
        Returns
        -------
        matplotlib.collections.LineCollection
            The generated line collection representing the colored line.
        """
        if "array" in lc_kwargs:
            warnings.warn('The provided "array" keyword argument will be overridden')
    
        # Check color array size (LineCollection still works, but values are unused)
        if len(c) != len(x) - 1:
            warnings.warn(
                "The c argument should have a length one less than the length of x and y. "
                "If it has the same length, use the colored_line function instead."
            )
    
        # Create a set of line segments so that we can color them individually
        # This creates the points as an N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, **lc_kwargs)
    
        # Set the values used for colormapping
        lc.set_array(c)
    
        return ax.add_collection(lc)

