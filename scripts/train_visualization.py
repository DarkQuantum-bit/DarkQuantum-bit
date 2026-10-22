"""Reproducible, illustrative MLP training animation (synthetic two-moons data)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.datasets import make_moons

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'assets' / 'ml-training.gif'

seed = 13
rng = np.random.default_rng(seed)
X, y = make_moons(n_samples=360, noise=0.18, random_state=seed)
X = (X - X.mean(0)) / X.std(0)
y = y[:, None].astype(float)

# Minimal two-layer neural network: 2 -> 16 (tanh) -> 1 (sigmoid).
params = {
    'W1': rng.normal(0, 0.35, (2,16)), 'b1': np.zeros((1,16)),
    'W2': rng.normal(0, 0.35, (16,1)), 'b2': np.zeros((1,1)),
}
m = {k:np.zeros_like(v) for k,v in params.items()}
v = {k:np.zeros_like(v) for k,v in params.items()}

def forward(x):
    hidden = np.tanh(x @ params['W1'] + params['b1'])
    logits = hidden @ params['W2'] + params['b2']
    probs = 1/(1+np.exp(-np.clip(logits,-30,30)))
    return hidden,probs

xs=np.linspace(-2.7,2.7,105)
ys=np.linspace(-2.2,2.2,85)
xx,yy=np.meshgrid(xs,ys)
mesh=np.c_[xx.ravel(),yy.ravel()]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig, (ax, loss_ax)=plt.subplots(1,2, figsize=(10.6,4.0), gridspec_kw={'width_ratios':[1.15,1]})
fig.patch.set_facecolor('#091923')
for a in (ax,loss_ax):
    a.set_facecolor('#0C2530')
    a.tick_params(colors='#95B6BE',labelsize=8)
    for sp in a.spines.values(): sp.set_color('#20434D')
fig.subplots_adjust(left=.06,right=.98,bottom=.15,top=.82,wspace=.27)
fig.text(.06,.935,'NEURAL NETWORK / LEARNING DYNAMICS',color='#70EDF3',fontsize=13,fontweight='bold')
fig.text(.06,.874,'Synthetic two-moons classification  •  MLP 2–16–1  •  Adam optimizer',color='#9AB9C2',fontsize=9)
frames=[]
losses=[]
steps=180
snapshots={0,1,2,3,5,7,10,14,19,26,35,47,62,80,104,133,160,180}
for t in range(steps+1):
    h,p=forward(X)
    eps=1e-8
    loss=float(-np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)))
    losses.append(loss)
    if t in snapshots:
        ax.clear();loss_ax.clear()
        for a in (ax,loss_ax):
            a.set_facecolor('#0C2530')
            a.tick_params(colors='#95B6BE',labelsize=8)
            for sp in a.spines.values():sp.set_color('#20434D')
        grid=forward(mesh)[1].reshape(xx.shape)
        ax.contourf(xx,yy,grid,levels=np.linspace(0,1,20),cmap=ListedColormap(['#113F50','#145062','#1B6372','#237B8A','#319AA5','#54C3CB','#8DE2E7','#B9F1F3','#D8FAFB','#E7FEFF'] ),alpha=.78)
        ax.contour(xx,yy,grid,levels=[.5],colors=['#EFFCFF'],linewidths=1.8)
        ax.scatter(X[y[:,0]==0,0],X[y[:,0]==0,1],s=13,c='#0A2436',edgecolors='#41D4E9',linewidths=.55)
        ax.scatter(X[y[:,0]==1,0],X[y[:,0]==1,1],s=13,c='#E2FEFF',edgecolors='#8CD9E4',linewidths=.4)
        ax.set_xlim(-2.7,2.7);ax.set_ylim(-2.2,2.2)
        ax.set_xlabel('Feature 1',color='#ACCCD1',fontsize=9)
        ax.set_ylabel('Feature 2',color='#ACCCD1',fontsize=9)
        ax.set_title('Decision boundary',color='#E1F8FA',fontsize=11,pad=10)
        loss_ax.plot(range(t+1),losses,c='#45DCE8',lw=2.1)
        loss_ax.scatter([t],[loss],c='#F0FEFF',s=28,zorder=3)
        loss_ax.set_xlim(0,steps);loss_ax.set_ylim(0.0,0.8)
        loss_ax.set_xlabel('Training step',color='#ACCCD1',fontsize=9)
        loss_ax.set_ylabel('Binary cross-entropy',color='#ACCCD1',fontsize=9)
        loss_ax.set_title('Training loss',color='#E1F8FA',fontsize=11,pad=10)
        loss_ax.grid(color='#28505B',alpha=.32)
        loss_ax.text(.96,.92,f'epoch {t:03d}  |  loss {loss:.3f}',transform=loss_ax.transAxes,ha='right',color='#A4EBF0',fontsize=9)
        fig.canvas.draw()
        frame=np.asarray(fig.canvas.buffer_rgba())[...,:3]
        frames.append(Image.fromarray(frame.copy()).convert('P',palette=Image.Palette.ADAPTIVE,colors=160))
    if t==steps:break
    # BCE gradients; full-batch Adam for reproducibility.
    dz2=(p-y)/len(X)
    grads={'W2':h.T@dz2,'b2':dz2.sum(axis=0,keepdims=True)}
    dh=(dz2@params['W2'].T)*(1-h*h)
    grads.update(W1=X.T@dh,b1=dh.sum(axis=0,keepdims=True))
    for k in params:
        m[k]=.9*m[k]+.1*grads[k]
        v[k]=.999*v[k]+.001*grads[k]**2
        mh=m[k]/(1-.9**(t+1));vh=v[k]/(1-.999**(t+1))
        params[k]-=.026*mh/(np.sqrt(vh)+1e-8)
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
frames[0].save(OUTPUT,save_all=True,append_images=frames[1:],duration=[650]+[260]*(len(frames)-2)+[1800],loop=0,optimize=False)
print(f'{OUTPUT}: {OUTPUT.stat().st_size/1024:.0f} KiB, loss {losses[0]:.3f} -> {losses[-1]:.3f}')
