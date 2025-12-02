import torch
#import matplotlib.pyplot as plt

def piecewise(start_temp=600,end_temp=300,total_rollouts=1000):
    tau = torch.linspace(0,1,total_rollouts)
    delT = start_temp - end_temp
    tau1=0.1
    tau2=0.8
    T1 = start_temp-delT*0.1
    T2 = start_temp-delT*0.5
    print(T1,T2)
    g = torch.zeros_like(tau)
    g = torch.where(tau<tau1,start_temp - delT*tau1*(tau/0.1),g)
    g = torch.where((tau<=tau2)&(tau>=tau1), T1-(T1-T2)*((tau-0.1)/0.7),g)
    g = torch.where(tau>tau2,(T2)-(T2-end_temp)*((tau-0.8)/0.2),g)

       
    return  g

#function= piecewise()
#plt.plot(function)
#plt.show()


