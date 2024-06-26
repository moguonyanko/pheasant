#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px

us_cities = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv")

'''
地図はブラウザで表示される。
参考:
https://plotly.com/python/mapbox-layers/
'''
def show_sample_scatter_mapbox():
  fig = px.scatter_mapbox(us_cities, 
                          lat="lat", lon="lon", 
                          hover_name="City", 
                          hover_data=["State", "Population"],
                          color_discrete_sequence=["fuchsia"], 
                          zoom=3, 
                          height=300)
  fig.update_layout(mapbox_style="open-street-map")
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
  fig.show()

def show_sample_scatter_mapbox_multi_mapbox_layers():
  fig = px.scatter_mapbox(us_cities, 
                          lat="lat", lon="lon", 
                          hover_name="City", 
                          hover_data=["State", "Population"],
                          color_discrete_sequence=["fuchsia"], 
                          zoom=3, 
                          height=300)
  fig.update_layout(
      mapbox_style="white-bg",
      mapbox_layers=[
          {
              "below": 'traces',
              "sourcetype": "raster",
              "sourceattribution": "United States Geological Survey",
              "source": [
                  "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
              ]
          },
          {
              "sourcetype": "raster",
              "sourceattribution": "Government of Canada",
              "source": ["https://geo.weather.gc.ca/geomet/?"
                        "SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX={bbox-epsg-3857}&CRS=EPSG:3857"
                        "&WIDTH=1000&HEIGHT=1000&LAYERS=RADAR_1KM_RDBR&TILED=true&FORMAT=image/png"],
          }
        ])
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
  fig.show()    

if __name__ == '__main__':
	print("gis module load")

# show_sample_scatter_mapbox()
show_sample_scatter_mapbox_multi_mapbox_layers()
