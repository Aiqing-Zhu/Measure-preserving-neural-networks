import argparse

import learner as ln
from LF_data import LFData

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='1')
args = parser.parse_args()

def main():
    
    device = 'cpu'# ’gpu' or 'cpu
    
    # data
    x0 = [0.1,1,1.1,0.5]
    h = 0.2
    train_num = 200
    test_num = 100
    data = LFData(x0, h, train_num, test_num)
  
    # training
    activation='sigmoid'
    net = ln.nn.VPNet(data.dim, h=0.01, s = int(data.dim/2), layers=4, sublayers=2, 
                       width=64, activation=activation)          

    arguments = {
        'filename': args.filename,
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': 0.001,
        'lr_decay': 1,
        'iterations': 2000,
        'batch_size': None,
        'print_every': 1000,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
    
    
    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
    
    
if __name__ == '__main__':
    
    main()
