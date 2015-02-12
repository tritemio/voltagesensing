"""
This module contains classes to add scroll bar to a Matplotlib figure.

This module works only when using the QT4 backend.
"""

from PyQt4 import QtGui, QtCore


class FrameScroller:
    """
    Add an horizontal scrollbar to a matplotlib figure to scroll through
    video frames.
    """
    def __init__(self, fig, image, video):
        # Save the variables
        self.fig = fig
        self.image = image
        self.video = video
        self.frame = 0

        # Save some MPL shortcuts
        self.draw = self.fig.canvas.draw
        self.draw_idle = self.fig.canvas.draw_idle
        self.ax = self.fig.axes[0]

        # Retrive the QMainWindow used by current figure and add a toolbar
        # to host the new widgets
        QMainWin = fig.canvas.parent()
        toolbar = QtGui.QToolBar(QMainWin)
        QMainWin.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        # Create the slider for scrolling frames
        self.set_slider(toolbar)

        # Setup the initial plot
        self.image.set_data(self.video[self.frame])

        # Draw text indicating the frame number
        self.text = self.ax.text(0.1,0.05, "Frame 0", transform=fig.transFigure)
        self.draw()

    def set_slider(self, parent):
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent=parent)
        self.slider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slider.setTickInterval(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.video.shape[0] - 1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(40)
        self.slider.setValue(0)  # set the initial position
        self.slider.valueChanged.connect(self.frame_changed)
        parent.addWidget(self.slider)

    def frame_changed(self, frame):
        self.image.set_data(self.video[frame])
        self.frame = frame
        self.text.set_text('Frame %d' % frame)
        self.draw()


class HorizontalScroller(object):
    """
    Add an horizontal scrollbar to a matplotlib figure to scroll
    along the X axis (i.e. time).
    """
    def __init__(self, fig, scroll_step=10, debug=False):
        # Setup data range variables for scrolling
        self.debug = debug
        if self.debug: print('HorizontalScroller init\n')

        self.fig = fig
        self.scroll_step = scroll_step
        self.xmin, self.xmax = fig.axes[0].get_xlim()
        self.width = 1 # axis units
        self.pos = 0   # axis units
        self.scale = 1e3 # conversion betweeen scrolling units and axis units

        # Some handy shortcuts
        self.ax = self.fig.axes[0]
        self.draw = self.fig.canvas.draw
        #self.draw_idle = self.fig.canvas.draw_idle

        # Retrive the QMainWindow used by current figure and add a toolbar
        # to host the new widgets
        QMainWin = fig.canvas.parent()
        toolbar = QtGui.QToolBar(QMainWin)
        QMainWin.addToolBar(QtCore.Qt.BottomToolBarArea, toolbar)

        # Create the slider and spinbox for x-axis scrolling in toolbar
        self.set_slider(toolbar)
        self.set_spinbox(toolbar)

        # Set the initial xlimits coherently with values in slider and spinbox
        self.ax.set_xlim(self.pos, self.pos + self.width)
        self.draw()

    def set_slider(self, parent):
        if self.debug:
            print('HorizontalScroller set_slider\n')
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, parent=parent)
        self.slider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slider.setTickInterval((self.xmax-self.xmin)/10.*self.scale)
        self.slider.setMinimum(self.xmin*self.scale)
        self.slider.setMaximum((self.xmax-self.width)*self.scale)
        self.slider.setSingleStep(self.width*self.scale/4.)
        self.slider.setPageStep(self.scroll_step*self.width*self.scale)
        self.slider.setValue(self.pos*self.scale) # set the initial position
        self.slider.valueChanged.connect(self.xpos_changed)
        parent.addWidget(self.slider)

    def set_spinbox(self, parent):
        if self.debug:
            print('HorizontalScroller set_spinbox\n')
        self.spinb = QtGui.QDoubleSpinBox(parent=parent)
        self.spinb.setDecimals(3)
        self.spinb.setRange(0.001,3600.)
        self.spinb.setSuffix(" s")
        self.spinb.setValue(self.width)   # set the initial width
        self.spinb.valueChanged.connect(self.xwidth_changed)
        parent.addWidget(self.spinb)

    def xpos_changed(self, pos):
        if self.debug:
            print("Position (in scroll units) %f\n" %pos)
        pos /= self.scale
        self.ax.set_xlim(pos, pos+self.width)
        self.draw()

    def xwidth_changed(self, width):
        if self.debug:
            print("Width (axis units) %f\n" % width)
        if width <= 0: return
        self.width = width
        self.slider.setSingleStep(self.width*self.scale/5.)
        self.slider.setPageStep(self.scroll_step*self.width*self.scale)
        old_xlim = self.ax.get_xlim()
        self.xpos_changed(old_xlim[0]*self.scale)

