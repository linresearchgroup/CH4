# Load library
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'ticks', color_codes = True)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#ignore warnings during runing models
import warnings
warnings.filterwarnings('ignore')

# Load datasets
ch4_ssa = pd.read_excel('CH4 SSA.xlsx').drop(['No'], axis = 1)
co2_ssa = pd.read_excel('CO2 SSA.xlsx').drop(['No'], axis = 1)
df1 = ch4_ssa.drop_duplicates()
df2 = co2_ssa.drop_duplicates()
df = df1.append(df2)

# Draw distplot of uptake amount
plt.figure(figsize = (8, 8))
plt.xlim(-1000, 5000)
plt.xlabel('SSA', fontsize = 18)
plt.ylabel('Percentage', fontsize = 18)
sns.distplot(df['SSA'], axlabel = 'Specific Surface Area (m2/g)', label = 'Percentage')
plt.savefig('02 SSA F1 Distplot.png', dpi = 300)
plt.show()

# Draw pairplot and heatmap
plt.figure(figsize = (8, 8))
sns.pairplot(df.drop(['SSA'], axis = 1), palette = 'husl', plot_kws=dict(edgecolor = None))
plt.savefig('02 SSA F2 Pairplot.png', dpi = 300)
plt.show()

plt.figure(figsize = (10, 8))
ax = sns.heatmap(df.drop(['SSA'], axis = 1).corr(), cmap="YlGnBu", annot = True, vmin = 0, vmax = 1)
ax.set_ylim([2, 0])
plt.savefig('02 SSA F3 Heatmap.png', dpi = 300)
plt.show()


for rs in [7, 14, 21, 35, 42, 84, 100, 120, 200, 420]:
    # Split the datasets into training and test datasets
    X = df.drop(['SSA'], axis = 1)
    y = df['SSA']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        shuffle = True, 
                                                        random_state = rs)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    # Standardization
    sds = StandardScaler()
    X_train_scaled = sds.fit_transform(X_train)
    X_test_scaled = sds.transform(X_test)
    
    with open('02 SSA Scaler RS{}.pickle'.format(rs), 'wb') as f:
         pickle.dump(sds, f)

    #'''
    # Multilayer Perceptron
    mlp = Sequential()
    mlp.add(Dense(units = 128, activation = 'relu', input_dim = X.shape[1]))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(units = 128, activation = 'relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(units = 64, activation = 'relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(units = 32, activation = 'relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(units = 1))
    mlp.compile(optimizer = 'adam', 
                loss='mean_squared_error', 
                metrics = ['mae'])
    
    mlp.fit(X_train_scaled, y_train, epochs = 2000, batch_size = 16)
    # Saving SSA model
    mlp.save('02 SSA RS{}.h5'.format(rs))


    y_pred_train = mlp.predict(X_train_scaled)
    y_pred_test = mlp.predict(X_test_scaled)

    y_pred_train = y_pred_train.reshape(-1)
    y_pred_test = y_pred_test.reshape(-1)

    train_array = pd.DataFrame({'y_train': np.array(y_train),
                                'y_pred_train': y_pred_train,
                                'AE':np.abs(y_train - y_pred_train),
                                'APE': np.abs((y_train - y_pred_train)/y_train)*100})
    test_array = pd.DataFrame({'y_test': np.array(y_test),
                               'y_pred_test':y_pred_test,
                               'AE': np.abs(y_test - y_pred_test),
                               'APE': np.abs((y_test - y_pred_test)/y_test)*100})
        
    train_array.to_excel('02 SSA RS{} train.xlsx'.format(rs))
    test_array.to_excel('02 SSA RS{} test.xlsx'.format(rs))       
   
    # Build DataFrame from evaluation metrics of both training and test datasets
    train_eval = pd.Series({'RMSE': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 2),
                            'R^2': round(r2_score(y_train, y_pred_train), 2),
                            'MAE': round(sum(train_array['AE'])/len(y_train), 2),
                            'MAPE':round(sum(train_array['APE'])/len(y_train), 2)})
    test_eval = pd.Series({'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2),
                            'R^2': round(r2_score(y_test, y_pred_test), 2),
                            'MAE': round(sum(test_array['AE'])/len(y_test), 2),
                            'MAPE':round(sum(test_array['APE'])/len(y_test), 2)})
    df2 = pd.DataFrame({'Train': train_eval, 
                        'Test': test_eval})
    # Draw figure for visualiation and comparison
    plt.figure(figsize = (8, 8))
    plt.scatter(y_train, y_pred_train, c='Tab:purple', alpha = 0.6, edgecolors = 'none',
                label= "Train (RMSE:{} R^2:{} MAE:{} MAPE:{})".format(df2.iloc[0, 0], df2.iloc[1, 0], df2.iloc[2, 0], df2.iloc[3,0]))
    plt.scatter(y_test, y_pred_test, c='Tab:blue', alpha = 0.6, edgecolors = 'none',
                label= "Test (RMSE:{} R^2:{} MAE:{} MAPE:{})".format(df2.iloc[0, 1], df2.iloc[1, 1], df2.iloc[2, 1], df2.iloc[3, 1]))
    plt.plot([0, 4000], [0, 4000], c ='orange', ls = '--')
    plt.axis([0, 4000, 0, 4000])
    plt.xlabel('y_test', fontsize = 18)
    plt.ylabel('y_pred', fontsize = 18)
    plt.title('02 SSA RS{}'.format(rs), fontsize = 22)
    plt.legend(loc="lower right", frameon = False)
    plt.savefig('02 SSA RS{}.png'.format(rs), dpi = 300)
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    