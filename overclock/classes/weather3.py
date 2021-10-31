# from pandas.io.formats import style
import requests
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

def get_sitelist():
    """ Return all weather stations circa 6000 with Met office api """
    url = "http://datapoint.metoffice.gov.uk/public/data/val/wxfcs/all/json/sitelist?&key=4961ab4f-8de7-46de-9311-c8baa03d84f3"
    r = requests.get(url).json()
    return pd.DataFrame(data=r['Locations']['Location']) 

def get_weather(id=98):
    """ get the weather for a location id """
    url = "http://datapoint.metoffice.gov.uk/public/data/val/wxfcs/all/json/" +str(id) + "?res=3hourly&key=4961ab4f-8de7-46de-9311-c8baa03d84f3"
    r = requests.get(url).json()
    return r
def lookup_wind_d(w_direc='N'):
    compass = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
    angle = np.linspace(0,15/16*360,16)
    return angle[compass.index(w_direc)]

def format_weather(weather,cols=True):
    date_day = []
    wind_d = [] # direction
    wind_s = [] # speed
    wind_g = [] # gust
    humid = []
    temperature = []
    rain_chance = []
    location_id = []
    latitude =[]
    longitude =[]
    elevation = []
    # could vectorise
    id = weather['SiteRep']['DV']['Location']['i'] 
    lat = weather['SiteRep']['DV']['Location']['lat']
    lon = weather['SiteRep']['DV']['Location']['lon']
    try:
        elev = weather['SiteRep']['DV']['Location']['elevation']
    except:
        elev = np.nan

    index = []
    idx = 0

    for p in weather['SiteRep']['DV']['Location']['Period']:
        dd = p['value']
        for r in p['Rep']:
            idx += 1
            location_id.append(id)
            index.append(idx) 
            wind_d.append(r['D'])
            wind_s.append(r['S'])
            wind_g.append(r['G'])
            rain_chance.append(r['Pp'])   
            date_day.append(dd)
            elevation.append(elev) 
            latitude.append(lat)
            longitude.append(lon)
            humid.append(r['H'])
            temperature.append(r['T'])
    if cols: 
        df = pd.DataFrame(data=zip(location_id,index,wind_d,wind_s,wind_g,rain_chance,date_day,latitude,longitude,elevation,humid,temperature),
        columns=["location_id","index","wind_d","wind_s","wind_g","rain_chance","date_day","latitude","longitude","elevation","humid","temperature"])
    else:
        df = pd.DataFrame(data=zip(location_id,index,wind_d,wind_s,wind_g,rain_chance,date_day,latitude,longitude,elevation,humid,temperature))
    return df


def all_weather():
    """
    Here, we return forecast for all sites.
    Params as per below
    'Param': [{'name': 'F', 'units': 'C', '$': 'Feels Like Temperature'}, {'name': 'G', 'units': 'mph', '$': 'Wind Gust'}, {'name': 'H', 'units': '%', 
    '$': 'Screen Relative Humidity'}, {'name': 'T', 'units': 'C', '$': 'Temperature'}, {'name': 'V', 'units': '', '$': 'Visibility'}, {'name': 'D', 'units': 'compass', '$': 'Wind Direction'}, {'name': 'S', 'units': 'mph', '$': 'Wind Speed'}, {'name': 'U', 'units': '', '$': 'Max UV Index'}, {'name': 'W', 'units': '', '$': 'Weather Type'}, {'name': 'Pp', 'units': '%', '$': 'Precipitation Probability'}]}, 'DV': {'dataDate': '2021-10-02T16:00:00Z'
    """
    df_sites = get_sitelist()
    df_subset = df_sites.iloc[0:-1:5,:]
    
    # wdf = df_subset.apply(lambda row : format_weather(get_weather(int(row['id']))), axis = 1) #,result_type='expand') # makes awkward, explode needed
    wdf = format_weather(get_weather(int(df_subset.iloc[0,:]['id'])))
    for _,row in df_subset.iloc[1:,:].iterrows():
        wdf = wdf.append(format_weather(get_weather(int(row['id'])))) # new obj needed, hence wdf = 
        # untested arg: ,cols=False
    return wdf

def main_example():
    """ Firstly, let's lookat sites, and then extract forecast for Warton """
    df_sites = get_sitelist()
    # print(df_sites)
    print('Show Warton, Coningsby and Lossie in sitelist')
    print(df_sites[df_sites.name == 'Warton'])
    print(df_sites[df_sites.name == 'Coningsby'])
    print(df_sites[df_sites.name == 'Lossiemouth'])
    
    weather = get_weather(int(df_sites[df_sites.name == 'Warton']['id']))    # 98
    dfw = format_weather(weather)
    print('Weather forecast for Warton, site id 98')
    print(dfw)

    print('Test quick lookup of true North as angle in °')
    a = lookup_wind_d(w_direc='N')
    print(a)

    print(" Ok, so let's try everything ")
    dfw = all_weather()    
    # dfw.columns = ["location_id","index","wind_d","wind_s","wind_g","rain_chance","date_day","latitude","longitude","elevation","humid","temperature"]
    dfw['wind_angle'] = dfw.apply(lambda row: lookup_wind_d(row['wind_d']), axis=1)
    dfw['wind_u'] = dfw['wind_s'].astype('float') * np.cos(dfw['wind_angle']) # x component hence cosine
    dfw['wind_v'] = dfw['wind_s'].astype('float') * np.sin(dfw['wind_angle'])
    dfw.to_csv(r'data\weather_data_full.csv', index = False)
    # print(dfw)
    
    

def gen_example_quiver():
    x,y = np.meshgrid(np.arange(-2, 2, .2),
                    np.arange(-2, 2, .25))
    z = x*np.exp(-x**2 - y**2)
    v, u = np.gradient(z, .2, .2)
    print(u,v)
    return x,y,u,v   

def mapbox(df,zfield = 'wind_s'):    
    fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='wind_s', radius=10,
                            center=dict(lat=55, lon=0), zoom=3,
                            mapbox_style="stamen-terrain",animation_frame="index")
    # fig.show()
    fig.write_html("outputs/mapbox.html")

def draw_quiv(x,y,u,v):

    # Create quiver figure
    fig = ff.create_quiver(x, y, u, v,#  color=np.arctan2(v/np.max(v), u/np.max(u)),
                        scale=.25,
                        arrow_scale=.4,
                        name='quiver',
                        line_width=1)

    # Add points to figure
    fig.add_trace(go.Scatter(x=[-0.171, -3.322], y=[53.094, 57.712],
                        mode='markers',
                        marker_size=12,
                        name='points'))
    """
    fig.add_scattergeo(lat = y# [53.094, 57.712]
                      ,lon = x# [-0.171, -3.322]
                      ,hoverinfo = 'none'
                      ,marker_size = 10
                      ,marker_color = 'rgb(65, 105, 225)' # blue
                      ,marker_symbol = 'star'
                      ,showlegend = False
                     )"""

    fig.update_geos(fitbounds="locations")
    # fig.show()
    fig.write_html("outputs/wind_vectors.html")
 
if __name__ == '__main__':
    main_example()
    df_all = pd.read_csv (r'data\weather_data_full.csv')
    df = df_all[df_all['index']==1]
    draw_quiv(df['longitude'],df['latitude'],df['wind_u']/np.max(df['wind_u']),df['wind_v']/np.max(df['wind_v']))
    # gen_example_quiver() # data only
    mapbox(df_all)
