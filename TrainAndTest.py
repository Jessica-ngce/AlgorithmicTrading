import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from arctic import Arctic
import pywt
import scaleogram as scg 

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


pd.options.mode.chained_assignment = None
plt.rcParams["figure.figsize"] = (18,5)




def clean_df(data):
    
    data = data.reset_index(drop=True)
    
    #return after 6 minutes
    data['return1'] = (data.c - data.c.shift(1))/data.c.shift(1)
    #return after 1 hour 
    #data['return10'] = (data.c - data.c.shift(10))/data.c.shift(10)
    #return after 1 day
    #data['return24'] = (data.c - data.c.shift(240))/data.c.shift(240)
    
    if data.return1.isna().sum() == 1:
        new = data.dropna(axis=0)
    else:
        new = data.filna(0)
    
    return new

def decomp_cwt(data):
    
    wavelet = "gaus5"
    scales = 2 ** np.arange(8)
    coef, freq = pywt.cwt(data['return1'], scales, wavelet)
    df_coef = pd.DataFrame(coef).T
    df_coef.columns = [str(int(i)) for i in 1/freq]
    
    for j in df_coef.columns:
        for i in range(1,10):
            df_coef[j + "_lag_" + str(i)] = df_coef[j].shift(i)
                
    df_coef = df_coef.fillna(0)
    return df_coef


def spect_cluster(data):

    X = np.array(data)
    X = StandardScaler().fit_transform(X)
    
    spectral = cluster.SpectralClustering(n_clusters = 4,
                                          eigen_solver = "arpack", affinity = "nearest_neighbors",
                                          assign_labels = "discretize", random_state = 30)
    
    spectral.fit(X)
    
    labels = spectral.labels_.astype(np.int)
    data['cluster'] = labels
    
    return data


def add_cluster(data):
    
    decompose_coef = decomp_cwt(data)
    coef_w_cluster = spect_cluster(decompose_coef)
    cluster_labels = coef_w_cluster['cluster']
    
    return cluster_labels


def get_result(data):
    
    clean = clean_df(data)
    clean['cluster']= add_cluster(clean).astype(str)
    clean = clean.dropna().reset_index()
    result = clean.groupby(['cluster']).agg({'return1': ['mean','std'], 'date': ['count'] , 'v':['mean','max']}).reset_index()
    result.columns = ['Clusters','Avg_Return','Risk','Duration_Min','Avg_Volume','Max_Volume']
    result['Duration_Min'] = result['Duration_Min']*6
    return (clean,result)

def pairs_cluster_res(data):
    
    pairs_df = []
    pairs_res = []
    
    for pairs, vals in data.groupby(['currency']):
        
        temp,temp_res = get_result(vals)
        temp_res['Currency'] = pairs
        pairs_df.append(temp)
        pairs_res.append(temp_res)
        
    pairs_df = pd.concat(pairs_df).drop('index',axis=1)
    pairs_res = pd.concat(pairs_res)
    
    return (pairs_df,pairs_res)
    


## Connect to MongoDB
conn = Arctic('127.0.0.1')
lib = conn['DE_project']
pairs7days = lib.read('currencies_7days').data




pairs7_cluster_df, pair7_cluster_res = pairs_cluster_res(pairs7days)
#lib.write('pairs7_cluster_df',pairs7_cluster_df) 
#lib.write('pair7_cluster_res',pair7_cluster_res) 



for pairs, vals in pairs7_cluster_df.groupby(['currency']):
    plt.plot(vals.c,label = pairs + ' closing price')
    plt.legend()
    plt.show()
    plt.plot(vals.return1,label = pairs + ' return1')
    plt.legend()
    plt.show()


print(pairs7_cluster_df.date.min(),pairs7_cluster_df.date.max())




for pairs,vals in pairs7_cluster_df.groupby(['currency']):
    print(pair7_cluster_res[pair7_cluster_res.Currency == pairs].set_index('Clusters'))
    print("\n")
    for i in range(4):
        plt.scatter(vals[vals.cluster == str(i)].index, vals.loc[vals.cluster == str(i),'c'], label = i)
    plt.plot()
    plt.title(pairs)
    plt.legend()
    plt.show()

## Load Currency Pairs from Polygon
def ts_to_datetime(ts): 
    temp = datetime.datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M') 
    return datetime.datetime.strptime(temp,'%Y-%m-%d %H:%M')


def currency_df(c_pair,begin_date,end_date):
    
    url = 'https://api.polygon.io/v2/aggs/ticker/C:{}/range/6/minute/{}/{}?adjusted=false&sort=asc&limit=50000&&apiKey=beBybSi8daPgsTp5yx5cHtHpYcrjp5Jq'.format(c_pair,begin_date,end_date)
    
    get_data = requests.get(url).json()
    results = pd.DataFrame(get_data['results'])
    results['currency'] = c_pair
    results['date'] = results.t.apply(lambda x: ts_to_datetime(x))
    results['weekday'] = results.date.apply(lambda x:x.weekday()+1)
    
    return results[['currency','c','v','date','h','l','o','weekday']]

def clean_features(data):
    
    data = data.reset_index(drop=True)
    ##return after 6 minutes
    data['return1'] = (data.c - data.c.shift(1))/data.c.shift(1)
    
    ##add trading range
    start_index = data.date.min(); end_index = data.date.max()
    time_indicies = pd.DataFrame(pd.date_range(start=start_index,end=end_index,freq='6min'),columns=['date'])
    new = time_indicies.merge(data,how = 'left',on='date')
    
    missing_vals = new.return1.isna().sum()
    ## drop na if # of NA is less than 1 pct of data
    if missing_vals < new.shape[0]*0.01:
        new = new.dropna()
    elif missing_vals <= new.shape[0]*0.1:
        
    ## replace na with KNN imputer 
        imputer = KNNImputer(n_neighbors = 3)
        new['return1'] = imputer.fit_transform(new[['return1']])
        new['v'] = imputer.fit_transform(new[['v']])
        new['c'] = imputer.fit_transform(new[['c']])
    
    else:
        raise Exception('Dataset contains larger than 10% of missing values.')
        
    new['currency'] = data.currency[1]
    new['weekday'] = new.weekday.apply(lambda x: str(x))
    
    ## remove noise
    new =  new[new.weekday != '6']
        
    return new

def get_data(begin_date,end_date):
    
    c = ['USDEUR','JPYKRW','GBPUSD','USDKRW','JPYEUR','USDJPY']

    l1 = []
    
    for i in range(len(c)):
        temp = currency_df(c[i],begin_date,end_date)
        clean = clean_features(temp)
        l1.append(clean)
    
    pairs = pd.concat(l1).reset_index(drop=True)
    return pairs


##################### get real time data####################################################################################
testing = get_data('2021-12-08','2021-12-10')


testing.currency.value_counts()




testing.currency.unique()


### Get data clusters and cluster results
def decomp_cwt(data):
    
    wavelet = "gaus5"
    scales = 2 ** np.arange(8)
    coef, freq = pywt.cwt(data['return1'], scales, wavelet)
    df_coef = pd.DataFrame(coef).T
    df_coef.columns = [str(int(i)) for i in 1/freq]
    
    for j in df_coef.columns:
        for i in range(1,10):
            df_coef[j + "_lag_" + str(i)] = df_coef[j].shift(i)
                
    df_coef = df_coef.fillna(0)
    return df_coef


def spect_cluster(data):

    X = np.array(data)
    X = StandardScaler().fit_transform(X)
    
    spectral = cluster.SpectralClustering(n_clusters = 4,
                                          eigen_solver = "arpack", affinity = "nearest_neighbors",
                                          assign_labels = "discretize", random_state = 30)
    
    spectral.fit(X)
    
    labels = spectral.labels_.astype(np.int)
    data['cluster'] = labels
    
    return data


def cwt_spec_cluster(data):
    
    decompose_coef = decomp_cwt(data)
    coef_w_cluster = spect_cluster(decompose_coef)
    cluster_labels = coef_w_cluster['cluster'].apply(lambda x:str(x))
    
    return cluster_labels

def get_result(data):
    
    data['cluster']= cwt_spec_cluster(data)
    data = data.dropna().reset_index()
    result = data.groupby(['cluster']).agg({'return1': ['mean','std'], 'date': ['count'] , 'v':['mean','max']}).reset_index()
    result.columns = ['Clusters','Avg_Return','Risk','Duration_Min','Avg_Volume','Max_Volume']
    result['Duration_Min'] = result['Duration_Min']*6
    return (data,result)


def pairs_cluster_res(data):
    
    pairs_df = []
    pairs_res = []
    
    for pairs, vals in data.groupby(['currency']):
        
        vals = vals.reset_index(drop=True)
        temp,temp_res = get_result(vals)
        temp_res['Currency'] = pairs
        pairs_df.append(temp)
        pairs_res.append(temp_res)
        
    pairs_df = pd.concat(pairs_df).drop('index',axis=1)
    pairs_res = pd.concat(pairs_res)
    
    return (pairs_df,pairs_res)



pairs3_cluster_df, pairs3_cluster_res=pairs_cluster_res(testing)

for pairs,vals in pairs3_cluster_df.groupby(['currency']):
    print(pairs3_cluster_res[pairs3_cluster_res.Currency == pairs].set_index('Clusters'))
    print("\n")
    for i in range(4):
        plt.scatter(vals[vals.cluster == str(i)].index, vals.loc[vals.cluster == str(i),'c'], label = i)
    plt.plot()
    plt.title(pairs)
    plt.legend()
    plt.show()