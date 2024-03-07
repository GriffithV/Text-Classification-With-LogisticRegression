import pandas as pd
from matplotlib import pyplot as plt

sample_data=pd.read_csv('adamloss.csv')

x=sample_data.iter
a=sample_data.w1
b=sample_data.w2
c=sample_data.w3
d=sample_data.w4
e=sample_data.w5   

plt.grid(True)


#plt.ylim(0,0.026)
#plt.xlim(0,10000)


plt.plot(x, a, color='red')
plt.plot(x+100, b, color='orange')
plt.plot(x+200, c, color='green')
plt.plot(x+300,d, color='blue')
plt.plot(x+400,e, color='purple')

plt.title("Loss/Epoch Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["W1=0.0", "W2=0.01","W3=-0.01","W4=0.03","W5=-0.03"], loc='upper right')
plt.show()
