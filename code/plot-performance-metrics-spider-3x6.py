
# https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from ..models.config import create_directory

dbs=[
    "chembl",
    ]
db_folders=[
    "./train-ch",
    ]

descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]
scalers=["minmax","std"]
models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc"]
splits=[1,2,3,4]
phases=["Train","Validation","Test"]

metrics=["precision","recall","accuracy","f1 score","auc","mcc"]
metric_names={"precision":"PRC","recall":"REC","accuracy":"ACC","f1 score":"F1","auc":"AUC","mcc":"MCC"}

folder=db_folders[0]

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def example_data(descriptor):
    for i in range(len(dbs)):
        db=dbs[i]
        folder=db_folders[i]

        df2=pd.DataFrame()
        for phase in phases:
            for enzyme in enzymes:
                for model in models:
                    print(db,enzyme,model)

                    path=folder+"/metrics/metrics-"+model+"-"+phase+".csv"
                    if(not os.path.exists(path)):
                        print("missing",path)
                        continue
                    df=pd.read_csv(path,delimiter=",",header=0,skipinitialspace=True)
                    #print(df.head())

                    md={}
                    md["Model"]=model.upper()
                    md["Phase"]=phase
                    md["Enzyme"]=enzyme
                    for metric in metrics:
                        mean=df[(df["enzyme"]==enzyme)&(df["descriptor"]==descriptor)][metric].mean()
                        mean=round(mean,4)
                        md[metric_names[metric]]=mean
                    #print(md)
                    df2=df2._append(md,ignore_index=True)

            df2.to_csv(folder+"/metrics-plots/performance-metrics-spider/tab - plot-metrics-010-spider-"+descriptor+".csv",index=False)
            df2.to_latex(folder+"/metrics-plots/performance-metrics-spider/tab - plot-metrics-010-spider-"+descriptor+".tex",float_format=lambda x: "%.4f" % x,index=False)

            for enzyme in enzymes:        
                df3=df2[df2["Enzyme"]==enzyme].copy()
                df3.drop(columns=["Enzyme"],inplace=True)
                df3.sort_values(by=["ACC"],ascending=False,inplace=True)
                df3.to_csv(folder+"/metrics-plots/performance-metrics-spider/tab - plot-metrics-020-spider-"+enzyme+"-"+descriptor+".csv",index=False)
                df3.to_latex(folder+"/metrics-plots/performance-metrics-spider/tab - plot-metrics-020-spider-"+enzyme+"-"+descriptor+".tex",float_format=lambda x: "%.4f" % x,index=False)

    x=[]
    #for phase in phases:
    x+=[model.upper() for model in models]
    data2=[x]
    #print(data2)

    # 3 phases
    for phase in phases:
        # 6 metrics
        for metric in metrics:
            # 3 enzymes
            l=[[],[],[]]
            k=0
            for enzyme in enzymes:
                l1=df2.loc[(df2["Enzyme"]==enzyme) & (df2["Phase"]==phase) ][metric_names[metric]].tolist()
                #print(metric,enzyme,l1)
                l[k]=l1
                k+=1
            
            #print(l)
            data2.append((metric_names[metric]+" "+phase,l))
    print(data2)

    return data2

if __name__ == '__main__':
    print("starting")
    N = 10
    theta = radar_factory(N, frame='polygon')

    create_directory(folder+"/metrics-plots/performance-metrics-spider/")
    for descriptor in descriptors:
        data = example_data(descriptor)
        spoke_labels = data.pop(0)

        fig, axs = plt.subplots(nrows=3,ncols=6,figsize=(14.4,7.2),dpi=600,subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.90, top=0.95, bottom=0.05)

        colors = ['b', 'r', 'g'] #, 'm', 'y']
        # Plot the four cases from the example data on separate Axes
        for ax, (title, case_data) in zip(axs.flat, data):
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
            ax.set_title(label=title, weight='bold', size='medium', loc='center', y=1.2, pad=2.0)
            for d, color in zip(case_data, colors):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
            ax.set_varlabels(spoke_labels)

        # add legend relative to top-left plot
        labels = ('IN', 'PR', 'RT')
        legend = axs[0, 0].legend(labels, loc=(0.99, 0.99), labelspacing=0.1, fontsize='small')

        plt.tight_layout()
        filename=folder+"/metrics-plots/performance-metrics-spider/fig - plot-metrics-030-spider-"+descriptor+"-3x6"
        plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=600)
        #plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=600)
        #plt.savefig(filename+".tif", bbox_inches="tight", orientation="portrait", dpi=600, format="tiff", pil_kwargs={"compression":"tiff_lzw"})
    print("stopping")
