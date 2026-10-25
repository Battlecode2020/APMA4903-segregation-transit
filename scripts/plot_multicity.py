"""Render a local scientific comparison figure from verified result tables."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from scripts.multicity_analysis import BASE, CITIES
from scripts.segregation_metrics import FOUR_CATEGORIES


def build():
    results=pd.read_csv(BASE/'city_comparison.csv')
    cities=[name for name in CITIES.values() if name in results.city.unique()]
    labels=['White alone','Black alone','Hispanic\n(any race)','Asian alone']
    fig,axes=plt.subplots(1,2,figsize=(12.5,8),layout='constrained')
    for ax,column,title in zip(axes,
        ['segregation_index_normalized','fryer_majority_ssi_minus_K'],
        ['Normalized PSSI: (λ − K) / (1 − K)','Adjusted Fryer: SSI − K']):
        values=results.pivot(index='city',columns='group',values=column).reindex(index=cities,columns=FOUR_CATEGORIES).to_numpy()
        norm=TwoSlopeNorm(vmin=min(-.01,float(values.min())),vcenter=0,vmax=max(.01,float(values.max())))
        image=ax.imshow(values,cmap='RdBu_r',norm=norm,aspect='auto')
        ax.set_xticks(range(4),labels,fontsize=10)
        ax.set_yticks(range(len(cities)),cities,fontsize=10)
        ax.set_title(title,fontsize=13,pad=14)
        ax.tick_params(length=0,pad=8)
        for i,j in np.ndindex(values.shape):
            rgba=image.cmap(norm(values[i,j]));brightness=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=10,color='white' if brightness<.5 else '#17283b')
        fig.colorbar(image,ax=ax,shrink=.65,label='Model score (dimensionless)')
    fig.suptitle('Eleven-city segregation estimates · 2020 city-limit tract populations',fontsize=15)
    fig.supxlabel('Rail/bus expected-time model + approximate walking; standardized $3 fare.\nDistance-only completion: Houston 49.3%, Phoenix 21.0%, Los Angeles 11.1% of pairs.\nFryer uses majority tracts; negative adjustments and zero-by-convention cases are retained.\nSchedules: October 20, 2026; San Francisco August 25, 2026. See coverage diagnostics.',fontsize=9)
    fig.savefig(BASE/'segregation_comparison.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':build()
