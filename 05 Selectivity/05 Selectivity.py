# Load library
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'ticks', color_codes = True)
from tensorflow.keras.models import load_model

#ignore warnings during runing models
import warnings
warnings.filterwarnings('ignore')

# Load MLP models
mlp_ch4 = load_model('01 CH4 Retrain.h5')
mlp_co2 = load_model('04 CO2 Retrain.h5')
mlp_ssa = load_model('02 SSA RS7.h5')

# Load Scaler model
with open('01 CH4 Retrain Scaler.pickle', 'rb') as f:
    sds_ch4 = pickle.load(f)
with open('04 CO2 Scaler.pickle', 'rb') as f:
    sds_co2 = pickle.load(f)
with open('02 SSA Scaler RS7.pickle', 'rb') as f:
    sds_ssa = pickle.load(f)

# Making the contour plot and meshing the grids
VMI = np.linspace(0.02, 1.0, 100)
VME = np.linspace(0.02, 1.0, 100)

# Create VMI and VME mesh for predicting ssa
grids = np.meshgrid(VMI, VME)
num = grids[0].reshape([-1]).shape[0]

ssa = np.zeros(shape = [num, 2])
ssa[:,0] = grids[0].reshape([-1])
ssa[:,1] = grids[1].reshape([-1])
ssa_scaled = sds_ssa.transform(ssa)
ssa_pred = mlp_ssa.predict(ssa)
ssa_pred = ssa_pred.flatten()

# Draw contour plot of SSA
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             ssa_pred.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 500)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of SSA', fontsize = 22)
plt.savefig('Contour Plot of SSA.png', dpi = 300)
plt.show()

# Create dataframe using generated ssa
x_gen_p1 = pd.DataFrame({'SSA': ssa_pred,
                      'VMI': ssa[:, 0],
                      'VME': ssa[:, 1],
                      'T': 25,
                      'P': 0.1})

x_gen_p9 = pd.DataFrame({'SSA': ssa_pred,
                      'VMI': ssa[:, 0],
                      'VME': ssa[:, 1],
                      'T': 25,
                      'P': 0.9})

x_gen_1 = pd.DataFrame({'SSA': ssa_pred,
                      'VMI': ssa[:, 0],
                      'VME': ssa[:, 1],
                      'T': 25,
                      'P': 1})

x_gen_5 = pd.DataFrame({'SSA': ssa_pred,
                      'VMI': ssa[:, 0],
                      'VME': ssa[:, 1],
                      'T': 25,
                      'P': 5})

x_gen_10 = pd.DataFrame({'SSA': ssa_pred,
                      'VMI': ssa[:, 0],
                      'VME': ssa[:, 1],
                      'T': 25,
                      'P': 10})

x_gen_20 = pd.DataFrame({'SSA': ssa_pred,
                      'VMI': ssa[:, 0],
                      'VME': ssa[:, 1],
                      'T': 25,
                      'P': 20})

# Predict CH4 uptake according to generated dataframe
# T25 P0.1
x_gen_ch4_p1 = sds_ch4.transform(x_gen_p1)
ch4_pred_p1 = mlp_ch4.predict(x_gen_ch4_p1)
ch4_pred_p1 = ch4_pred_p1.reshape(-1)
# Draw contour plot of CH4 uptake
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             ch4_pred_p1.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 500)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of CH4 T25P0.1', fontsize = 22)
plt.savefig('Contour Plot of CH4 T25P0.1.png', dpi = 300)
plt.show()
ch4_t25pp1 = x_gen_p1
ch4_t25pp1['ch4_t25p0.1'] = ch4_pred_p1
ch4_t25pp1.to_excel('ch4_t25p0.1.xlsx')

# T25 P1
x_gen_ch4_1 = sds_ch4.transform(x_gen_1)
ch4_pred_1 = mlp_ch4.predict(x_gen_ch4_1)
ch4_pred_1 = ch4_pred_1.reshape(-1)
# Draw contour plot of CH4 uptake
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             ch4_pred_1.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 500)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of CH4 T25P1', fontsize = 22)
plt.savefig('Contour Plot of CH4 T25P1.png', dpi = 300)
plt.show()
ch4_t25p1 = x_gen_1
ch4_t25p1['ch4_t25p1'] = ch4_pred_1
ch4_t25p1.to_excel('ch4_t25p1.xlsx')

# T25 P5
x_gen_ch4_5 = sds_ch4.transform(x_gen_5)
ch4_pred_5 = mlp_ch4.predict(x_gen_ch4_5)
ch4_pred_5 = ch4_pred_5.reshape(-1)
# Draw contour plot of CH4 uptake
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             ch4_pred_5.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 500)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of CH4 T25P5', fontsize = 22)
plt.savefig('Contour Plot of CH4 T25P5.png', dpi = 300)
plt.show()
ch4_t25p5 = x_gen_5
ch4_t25p5['ch4_t25p5'] = ch4_pred_5
ch4_t25p5.to_excel('ch4_t25p5.xlsx')

# T25 P10
x_gen_ch4_10 = sds_ch4.transform(x_gen_10)
ch4_pred_10 = mlp_ch4.predict(x_gen_ch4_10)
ch4_pred_10 = ch4_pred_10.reshape(-1)
# Draw contour plot of CH4 uptake
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             ch4_pred_10.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 500)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of CH4 T25P10', fontsize = 22)
plt.savefig('Contour Plot of CH4 T25P10.png', dpi = 300)
plt.show()
ch4_t25p10 = x_gen_10
ch4_t25p10['ch4_t25p10'] = ch4_pred_10
ch4_t25p10.to_excel('ch4_t25p10.xlsx')

# T25 P5
x_gen_ch4_20 = sds_ch4.transform(x_gen_20)
ch4_pred_20 = mlp_ch4.predict(x_gen_ch4_20)
ch4_pred_20 = ch4_pred_20.reshape(-1)
# Draw contour plot of CH4 uptake
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             ch4_pred_20.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 500)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of CH4 T25P20', fontsize = 22)
plt.savefig('Contour Plot of CH4 T25P20.png', dpi = 300)
plt.show()
ch4_t25p20 = x_gen_20
ch4_t25p20['ch4_t25p20'] = ch4_pred_20
ch4_t25p20.to_excel('ch4_t25p20.xlsx')

# Predict CO2 uptake according to generated dataframe
# T25P0.9
x_gen_co2_p9 = sds_co2.transform(x_gen_p9)
co2_pred_p9 = mlp_co2.predict(x_gen_co2_p9)
co2_pred_1 = co2_pred_p9.reshape(-1)
# Draw contour plot of CO2 uptake
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             co2_pred_p9.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 50)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of CO2 T25P0.9', fontsize = 22)
plt.savefig('Contour Plot of CO2 T25P0.9.png', dpi = 300)
plt.show()
co2_t25pp9 = x_gen_p9
co2_t25pp9['co2_t25p0.9'] = co2_pred_p9
co2_t25pp9.to_excel('co2_t25p0.9.xlsx')

# T25P1
x_gen_co2_1 = sds_co2.transform(x_gen_1)
co2_pred_1 = mlp_co2.predict(x_gen_co2_1)
co2_pred_1 = co2_pred_1.reshape(-1)
# Draw contour plot of CO2 uptake
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             co2_pred_1.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 50)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of CO2 T25P1', fontsize = 22)
plt.savefig('Contour Plot of CO2 T25P1.png', dpi = 300)
plt.show()
co2_t25p1 = x_gen_1
co2_t25p1['co2_t25p1'] = co2_pred_1
co2_t25p1.to_excel('co2_t25p1.xlsx')


# Selectivity
selectivity = pd.DataFrame({'VMI': ssa[:, 0],
                            'VME': ssa[:, 1]})
# CH4 T25P1 CO2 T25P1
select_h1_o1 = co2_pred_1 / ch4_pred_1
# Draw contour plot of selectivity T25P1
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             select_h1_o1.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 50)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of Selectivty CH4_CO2 T25P1', fontsize = 22)
plt.savefig('Contour plot of Selectivty CH4_CO2 T25P1.png', dpi = 300)
plt.show()
select_t25p1 = Selectivity
select_t25p1['select_h1_o1'] = select_h1_o1
select_t25p1.to_excel('select_h1_o1.xlsx')

# CH4 T25P0.9 CO2 T25P0.1
select_hp9_op1 = co2_pred_p1 / ch4_pred_p9
# Draw contour plot of selectivity T25P1
plt.figure(figsize = (10, 8))
plt.contourf(VMI, 
             VME, 
             select_hp9_op1.reshape(grids[0].shape), 
             cmap = 'jet', 
             levels = 50)
plt.xlabel('VMI', fontsize = 18)
plt.ylabel('VME', fontsize = 18)
plt.colorbar()
plt.title('Contour plot of Selectivty CH4_CO2 T25P1', fontsize = 22)
plt.savefig('Contour plot of Selectivty CH4_CO2 T25P1.png', dpi = 300)
plt.show()
select_t25pp1 = Selectivity
select_t25pp1['select_h0.9_o0.1'] = select_hp9_op1
select_t25pp1.to_excel('select_h0.9_o0.1.xlsx')























