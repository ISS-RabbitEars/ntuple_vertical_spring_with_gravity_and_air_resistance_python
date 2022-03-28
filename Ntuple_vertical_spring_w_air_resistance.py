import sys
import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm

def integrate(ic, ti, p):
	ic_list = ic
	m, k, yeq, gc, rho, cd, ar = p
	
	y = []
	v = []
	for i in range(m.size):
		y.append(ic_list[2 * i])
		v.append(ic_list[2 * i + 1])

	sub = {}
	for i in range(m.size):
		sub[M[i]] = m[i]
		sub[K[i]] = k[i]
		sub[YEQ[i]] = yeq[i]
		sub[Y[i]] = y[i]
		sub[Ydot[i]] = v[i]
		sub[Ar[i]] = ar[i]
	sub['g'] = gc
	sub['RHO'] = rho
	sub['CD'] = cd

	diff_eq = []
	for i in range(m.size):
		diff_eq.append(v[i])
		diff_eq.append(A[i].subs(sub))

	print(ti)

	return diff_eq

#---SymPy Derivation (also sets N parameter - number of masses/springs)

N = 3

RHO, CD, g, t = sp.symbols('RHO CD g t')
Ar = sp.symbols('Ar0:%i' %N)
M = sp.symbols('M0:%i' %N)
K = sp.symbols('K0:%i' %N)
YEQ = sp.symbols('YEQ0:%i' %N)
Y = dynamicsymbols('Y0:%i' %N)

Ydot = []
T = 0
for i in range(N):
	Ydot.append(Y[i].diff(t, 1))
	T += M[i] * Ydot[i]**2
T *= sp.Rational(1, 2)

V1 = K[0] * (Y[0] - YEQ[0])**2
V2 = M[0] * Y[0]
for i in range(1, N):
	V1 += K[i] * (Y[i] - Y[i-1] - YEQ[i])**2
	V2 += M[i] * Y[i]
V = sp.Rational(1, 2) * V1 + g * V2

L = T - V

A = []
Fc = sp.Rational(1, 2) * RHO * CD
for i in range(N):
	dLdY = L.diff(Y[i], 1)
	dLdYdot = L.diff(Ydot[i], 1)
	ddtdLdYdot = dLdYdot.diff(t, 1)
	F = Fc * Ar[i] * sp.sign(Ydot[i]) * Ydot[i]**2
	dL = ddtdLdYdot - dLdY + F
	sol = sp.solve(dL, Y[i].diff(t, 2))
	A.append(sp.simplify(sol[0]))

#--------------------------------------------------------

#----parameters, SciPy integration (intgration function at top)
#----and energy calculations----------------------------------

gc = 9.8
mass_a, mass_b = [1, 2]
k_a, k_b = [50, 100]
yeq_a, yeq_b = [-1, -1]
yo_a, yo_b = [-1, -3]
vo_a, vo_b = [0, 0]
rho = 1.225
cd = 0.47
rad = 0.25
tf = 60 
nfps = 30


initialize = "increment"


if initialize == "increment":
	m = np.linspace(mass_a, mass_b, N)
	k = np.linspace(k_a, k_b, N)
	yeq = np.linspace(yeq_a, yeq_b, N)
	yo = np.linspace(yo_a, yo_b, N)
	vo = np.linspace(vo_a, vo_b, N)
elif initialize == "random":
	rng=np.random.default_rng(92314311)
	m = (mass_b - mass_a) * np.random.rand(N) + mass_a
	k = (k_b - k_a) * np.random.rand(N) + k_a
	yeq = (yeq_b - yeq_a) * np.random.rand(N) + yeq_a
	yo = (yo_b - yo_a) * np.random.rand(N) + yo_a
	vo = (vo_b - vo_a) * np.random.rand(N) + vo_a
else:
	sys.exit("Initialization Routine Not Found. Choices are increment or random. Pick One.")

mass_radius = "proportional"
mr = np.zeros(N)
if mass_radius == "uniform":
        mr[:] = rad
elif mass_radius == "proportional":
        mr[:] = rad*m[:]/max(m)
else:
        sys.exit("Mass Radius Initialization Routine Not Found. Choices are uniform or proportional. Pick One.")
ar = np.pi * mr**2


p = [m, k, yeq, gc, rho, cd, ar]
ic = []
for i in range(N):
	ic.append(yo[i])
	ic.append(vo[i])

nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

yv = odeint(integrate, ic, ta, args=(p,))

y = np.zeros((N, nframes))
ke = np.zeros(nframes)
pe = np.zeros(nframes)
for i in range(nframes):
	ke_sub={}
	pe_sub={}
	for j in range(N):
		y[j][i] = yv[i, 2 * j]
		ke_sub[M[j]] = m[j]
		ke_sub[Ydot[j]] = yv[i, 2 * j + 1]
		pe_sub[M[j]] = m[j]
		pe_sub[K[j]] = k[j]
		pe_sub[YEQ[j]] = yeq[j]
		pe_sub[Y[j]] = y[j][i]
	pe_sub['g'] = gc
	ke[i] = T.subs(ke_sub)
	pe[i] = V.subs(pe_sub)

E = ke + pe

#----aesthetics, plot, animation---------------

fig, a = plt.subplots()

ymin = y.min() 
ymax = 0
xline = 0
ymax += 2*mr[0]
ymin -= 2*mr[N-1]
xmax = xline + 2 * max(mr)
xmin = xline - 2 * max(mr)

spring_constant_proportional = "y"
dl = np.zeros((N,nframes))
nl = np.zeros(N)
dl[0][:] = np.abs(y[0][:] - mr[0])
nl[0] = int(np.ceil((max(dl[0]))/(2 * mr[0])))
for i in range(1,N):
	dl[i][:] = np.abs(y[i][:] - y[i-1][:])
	nl[i] = int(np.ceil(max(dl[i])/(2 * mr[i])))
lsf = 1
if spring_constant_proportional == "y":
	lr = np.zeros(N)
	lr[:] = 1 / (k[:] / max(k))
	for i in range(N):
		nl[i] = int(lsf * lr[i] * nl[i])

nlmax = int(max(nl))
xl = np.zeros((N,nlmax,nframes))
yl = np.zeros((N,nlmax,nframes))
for i in range(nframes):
	l0 = np.abs((y[0][i]/nl[0]))
	yl[0][0][i] = y[0][i] + mr[0] + 0.5*l0
	for k in range(1,int(nl[0])):
		yl[0][k][i] = yl[0][k-1][i] + l0
	for k in range(int(nl[0])):
		xl[0][k][i] = xline+((-1)**k)*(np.sqrt(mr[0]**2 - (0.5*l0)**2))
	for j in range(1,N):
		lj = (np.abs(y[j][i]-y[j-1][i])-(mr[j]+mr[j-1]))/nl[j]
		yl[j][0][i] = y[j][i] + mr[j] + 0.5*lj
		for k in range(1,int(nl[j])):
			yl[j][k][i] = yl[j][k-1][i] + lj
		for k in range(int(nl[j])):
			xl[j][k][i] = xline+((-1)**k)*(np.sqrt(mr[j]**2 - (0.5*lj)**2))

clist = cm.get_cmap('gist_rainbow', N)
			
def run(frame):
	plt.clf()
	plt.subplot(181)
	for i in range(N):
		circle=plt.Circle((xline,y[i][frame]),radius=mr[i],fc=clist(i))
		plt.gca().add_patch(circle)
	plt.plot([xl[0][int(nl[0])-1][frame],xline],[yl[0][int(nl[0])-1][frame],mr[0]],'xkcd:cerulean')
	for i in range(N):
		plt.plot([xline,xl[i][0][frame]],[y[i][frame]+mr[i],yl[i][0][frame]],'xkcd:cerulean')
	for i in range(1,N):
		plt.plot([xl[i][int(nl[i])-1][frame],xline],[yl[i][int(nl[i])-1][frame],y[i-1][frame]-mr[i-1]],'xkcd:cerulean')
	for j in range(N):
		for i in range(int(nl[j])-1):
			plt.plot([xl[j][i][frame],xl[j][i+1][frame]],[yl[j][i][frame],yl[j][i+1][frame]],'xkcd:cerulean')
	plt.title("N-Tuple Vertical\nSpring with Air\nResistance (N=%i)" %N)
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(1,8,(2,8))
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')
	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('ntuple_vertical_spring_w_air_resistance.mp4', writer=writervideo)
plt.show()



 



