import numpy as np
import geopandas as gpd

pc = gpd.read_file('pc.shp')
pc['y'] = pc.Idcar_200m.str.slice(15,).str.split('E', expand=True).astype(int).loc[:,0]
pc['x'] = pc.Idcar_200m.str.slice(15,).str.split('E', expand=True).astype(int).loc[:,1]
pc.y = (pc.y - pc.y.min())/200 
pc.x = (pc.x - pc.x.min())/200 
pc = pc[pc.x.isin(np.arange(11, 139))&pc.y.isin(np.arange(17,145))]

pc['i'] = pc.x - pc.x.min() 
pc['j'] = pc.y - pc.y.min() 
pc = pc[['Idcar_200m','i','j', 'Men', 'Men_pauv', 'geometry']]
pc.i = pc.i.astype(int)
pc.j = pc.j.astype(int)

pc['woh'] = pc.Men - pc.Men_pauv
pc = pc.rename(columns={'Men_pauv': 'ph'})

aph = np.zeros((128,128))
awoh = np.zeros((128,128))
aph[127-pc.j.values,pc.i.values] = pc.ph
awoh[127-pc.j.values,pc.i.values] = pc.woh

apc = np.stack((aph,awoh), axis=-1)
np.save('paris_household.npy', apc)
