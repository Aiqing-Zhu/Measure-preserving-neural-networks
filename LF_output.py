import torch
import matplotlib.pyplot as plt

from LF_data import LFData


def main():
    fig, ax= plt.subplots(1,2, figsize=(14,10))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.01, hspace=0.2)
    x0 = [0.1,1,1.1,0.5]
    e=200
    d=100
    
    data = LFData(x0, 0.2, 2, 2)
    flow_true = data.solver.flow(data.X_train_np[0], data.h, e+d)
    df = int((d)*data.h/0.01)+1
    ef = int((e)*data.h/0.01)+1
    
    flow= data.solver.flow(x0, 0.01, ef+df)
    

    x,y=0,1 #The ploted component, can be choosen from 0, 1, 2 and 3
    ax[0].plot(flow[0:ef,x],flow[0:ef,y], linestyle= '--',
                  color = 'grey', linewidth =1, label='Ground truth',zorder = 0)


    ax[0].scatter(flow_true[0,x],flow_true[0,y], 
                color = 'red', s = 30, label='Start point',marker = 'v', zorder = 2)    
    ax[0].scatter(flow_true[e,x],flow_true[e,y], 
                color = 'orange', s = 30, label='End point',marker = '^', zorder = 2)
    ax[0].scatter(flow_true[:e+1,x],flow_true[:e+1,y], 
                color = 'grey', s = 20, label='Observation',zorder = 0)     
    predict(ax[0], data, flow_true, 0,e+1, label = 'Reconstruction',
            net_type='g',color='green', marker='+', zorder =1)
    
    
    ef = int((e)*data.h/0.01)
    df = int((d-1)*data.h/0.01)
    ax[1].scatter(flow_true[e,x],flow_true[e,y], 
                color = 'orange', s = 30, label='Start point',marker = '^', zorder = 2)
    ax[1].plot(flow[ef:ef+df,x],flow[ef:ef+df,y], linestyle= '--',
                 color = 'grey', linewidth =1, label='Ground truth',zorder = 0)
    ax[1].scatter(flow_true[e:e+d,x],flow_true[e:e+d,y], 
                    color = 'grey', s=20, label='Exact position',zorder = 0)   
    predict(ax[1], data, flow_true, e,d, label = 'Prediction',
            net_type='g',color='blue', marker='+', zorder =1)


    xsize=18
    legendsize=18
    titlesize=22
    
    a,b =-1.55,1.55      
    ax[0].set_xlim(a,b)
    ax[1].set_xlim(a,b)
    ax[0].set_ylim(a,2.85)
    ax[1].set_ylim(a,2.85)
    
    ax[0].legend(loc='upper right',fontsize=legendsize)
    ax[1].legend(loc='upper right',fontsize=legendsize)    

    ax[0].set_title('Reconstruction',fontsize=titlesize,loc='left')
    ax[1].set_title('Prediction',fontsize=titlesize,loc='left')
    ax[0].set_xlabel(r'$x_1$', fontsize=xsize)
    ax[1].set_xlabel(r'$x_1$', fontsize=xsize)
    ax[0].set_ylabel(r'$x_2$', fontsize=xsize)
    
    
    ax[1].set_yticks([])
    ax[0].set_yticks([-1,0,1,2])
    ax[0].set_xticks([-1,0,1])
    ax[1].set_xticks([-1,0,1])

    fig.savefig('LF.pdf',dpi=300, bbox_inches='tight')
    return 0    
def predict(ax1, data, flow_true, e, d, net_type='l',label = 'Prediction', color='red', marker='x', zorder=1):

    n = '1' # the filename in main.py
    net = torch.load('outputs/'+n+'/model_best.pkl', map_location=torch.device('cpu'))
    flow_pred = net.predict(torch.FloatTensor(flow_true[e]), d, keepinitx=True, returnnp=True)

    x,y =0,1
    ax1.scatter(flow_pred[:d,x],flow_pred[:d,y], 
                color=color, marker=marker, label=label, zorder=zorder)    
    return 0

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    