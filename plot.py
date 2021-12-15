import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=[20, 12])

plt.xlim([-1.7, 2.2])
plt.ylim([-1.2, 1.05])

a, b =1.7,1.
c = np.linspace(0, 2*np.pi, 1000)
x,y = a*np.cos(c), b*np.sin(c)
plt.plot(x,y,color = 'black',linewidth=2,zorder=1)

the = 0.5
xx=1.9
x = np.array([-1.5, -0.7, 0.5, a*np.cos(the), xx])
y = np.array([0.1, -0.7, -0.4, b*np.sin(the), 
             a*np.sin(the)/(b*np.cos(the))*(xx - a*np.cos(the))+b*np.sin(the)])
plt.scatter(x, y, color='red', marker='o', linewidths=10, zorder=2)

plt.plot(x[1:5],y[1:5], linestyle='--', zorder=1)
plt.plot([x[1],x[4]],[y[1],y[4]], linestyle='--', zorder=1)


fontsize1 = 35
plt.text(-0.8, 0.5, r'Hypothesis space', fontsize=fontsize1, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='center')
plt.text(x[0], y[0]-0.07, r'$f_{init}$', fontsize=fontsize1, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text(x[1], y[1]-0.07, r'$f_{net}$', fontsize=fontsize1, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text(x[2], y[2]-0.07, r'$f_{min}$', fontsize=fontsize1, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text(x[3], y[3]-0.1, r'$f^*$', fontsize=fontsize1, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text(x[4]+0.1, y[4]-0.07, r'$f_{tag}$', fontsize=fontsize1, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')


plt.annotate("",
            xy=(x[1], y[1]+0.04),
            xytext=(x[0], y[0]),
            size=10, va="center", ha="center",
            arrowprops=dict(color='gray',
                            width=0.1,
                            headlength=7,headwidth=10,
                            connectionstyle="arc3,rad=-0.4",
                            ),
            zorder=1)
fontsize2 = 35            
plt.text((x[0]+x[1])/2+0.05, (y[0]+y[1])/2+0.1, 'Training', fontsize=30, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text((x[1]+x[2])/2, (y[1]+y[2])/2-0.05, r'$OE$', fontsize=fontsize2, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text((x[2]+x[3])/2, (y[2]+y[3])/2-0.07, r'$GE$', fontsize=fontsize2, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text((x[3]+x[4])/2+0.04, (y[3]+y[4])/2-0.04, r'$AE$', fontsize=fontsize2, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')
plt.text((x[1]+x[4])/2, (y[1]+y[4])/2-0.05, r'$EE$', fontsize=fontsize2, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='top')


plt.text(0.3,-1.2, r'$AE \leq EE \leq AE + GE + OE$', fontsize=30, color='black', zorder=2,
          horizontalalignment='center', verticalalignment='bottom')


plt.xticks([])
plt.yticks([])
plt.axis('off')



plt.savefig('error_types.pdf', dpi=500, bbox_inches='tight')
plt.show()
