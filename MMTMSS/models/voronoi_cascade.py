import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import voronoi_diagram
from shapely.geometry import Polygon, Point

rng = np.random.default_rng()

def split_in_voronoi(polygon, sigma1, sigma2, corr, pts = 4):
    
    test = False
    while not(test):   
        x = rng.uniform(polygon.bounds.minx[0],polygon.bounds.maxx[0], pts * 3)
        y = rng.uniform(polygon.bounds.miny[0],polygon.bounds.maxy[0], pts * 3)
        points  = gpd.GeoSeries([Point(xy) for xy in zip(x,y)])
        points = points[points.intersects(polygon.geometry[0])]
        if points.shape[0] >= pts:
            test=True
            points = points[:pts].unary_union    
            
    data = np.exp(rng.multivariate_normal(np.array([0,0]),  np.array([[sigma1**2,corr*sigma1*sigma2], [corr*sigma1*sigma2, sigma2**2]]), 4))
    data = pd.DataFrame(data, columns=['P', 'Q'])
    data.P = polygon.P[0] * data.P
    data.Q = polygon.Q[0] * data.Q
    data['id'] = np.char.add(np.repeat(polygon.id[0]+'_',pts), np.arange(0,pts).astype(str))
    
    
    new_polygons = gpd.GeoDataFrame(data, geometry=gpd.GeoSeries(voronoi_diagram(points, envelope=polygon.unary_union)).explode(index_parts=True).intersection(polygon.unary_union).reset_index( drop=True))#.reset_index(drop=True)
    return(new_polygons)


def recurence(n, polygons,listespolygons, sigma1, sigma2, corr):
    listespolygons[n-1].append(polygons)
    if n > 1:
        for i in range(polygons.shape[0]):            
            recurence(n-1,split_in_voronoi(polygons[i:(i+1)].reset_index([0], drop=True),sigma1, sigma2, corr),
                      listespolygons,
                      sigma1, 
                      sigma2, 
                      corr)

        
def voronoi_cascade(sigma1=.3, sigma2=.3, corr=-0.8, n=3):
    """
    Compute a vornoi multifractal cascade. It's an extension of classic multifractal cascade. 
    At each step, instead of share weights in a dyadic way, 4 random points are sampled in each polygon.
    Based on this points a Voronoi diagram is computed, and the weights are shared betweem them. 
             

    Parameters
    ----------
    sigma1 : float, optional
        Standart deviation of P. The default is .3.
    sigma2 : TYPE, optional
        DESCRIPTION. The default is .3.
    corr : TYPE, optional
        DESCRIPTION. The default is -0.8.
    n : int, optional
        Number of steps (scale).

    Returns
    -------
    A Dictionary of GeoDataFrames. The index is the step. 
    """
    data = np.exp(rng.multivariate_normal(np.array([0,0]),  np.array([[sigma1**2,corr*sigma1*sigma2], [corr*sigma1*sigma2, sigma2**2]]), 1))
    data = pd.DataFrame(data, columns=['P', 'Q'])
    data['id'] = '0'
    polygon = gpd.GeoDataFrame(data,geometry=gpd.GeoSeries(Polygon([(0,0), (0,1), (1,1), (1,0)])))
    
    listespolygons = dict()
    for i in range(n):
        listespolygons[i] = []

    recurence(n, polygon, listespolygons, sigma1, sigma2, corr)

    voronoi = dict()
    for i in range(n):
        voronoi[n-i-1] = pd.concat(listespolygons[i])
    return(voronoi)
