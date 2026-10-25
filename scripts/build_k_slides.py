"""Build self-contained Overleaf sources with editable PGFPlots figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'outputs/k_approximation'; S=ROOT/'slides/k_approximation'
toy=pd.read_csv(O/'toy_summary.csv'); large=pd.read_csv(O/'large_summary.csv'); meta=json.loads((O/'results.json').read_text())
frames=[]
def frame(title,body):frames.append('\\begin{frame}{'+title+'}\n\\small\n'+body+'\n\\end{frame}\n')
def coords(x,y):return ' '.join(f'({a:.9g},{b:.9g})' for a,b in zip(x,y))
def line(x,y,opt,label=None):
    return '\\addplot['+opt+'] coordinates {'+coords(x,y)+'};\n'+('\\addlegendentry{'+label+'}\n' if label else '')
def axis(body,xlabel,ylabel,extra=''):
    return r'\begin{center}\begin{tikzpicture}\begin{axis}[width=12.6cm,height=5.05cm,grid=major,grid style={gray!15},tick label style={font=\scriptsize},label style={font=\small},legend style={font=\scriptsize,draw=none,fill=white},legend pos=north west,xlabel={'+xlabel+'},ylabel={'+ylabel+'},'+extra+']\n'+body+r'\end{axis}\end{tikzpicture}\end{center}'
frame(r'The $K$ approximation and its random-allocation target',r'''
For neighborhood populations $n_i$, total population $N=\sum_i n_i$, and social affinities $f(i\to j)$,
\[
 P_{ij}=\frac{f(i\to j)n_j}{\sum_\ell f(i\to\ell)n_\ell},\qquad
 A=P\operatorname{diag}(r),\qquad \Lambda=\rho(A).
\]
Fix $M$ group labels and assign them uniformly to the $N$ resident slots.
Write $K=M/N$ and $r_i=B_i/n_i$ after assignment.
\[
 m_0(P,K)=\mathbb E_0[\Lambda],\qquad
 \mathbb E_0[r_i]=K,\qquad \rho(P\operatorname{diag}(\mathbb E_0r))=K.
\]
\begin{block}{The question}
Does $m_0(P,K)\approx K$? The spectral radius is nonlinear, so taking the expectation inside it is not generally valid.
\end{block}
This section uses the project's global PSSI spectral radius, not the component-averaged Fryer SSI.
''')
frame(r'Equal-strength complete mixing gives $m_0=K$ exactly',r'''
Assume every ordered pair has the same positive affinity, including co-location:
$f(i\to j)=c>0$. Then $P=\mathbf1w^T$, where $w_j=n_j/N$.
\[
 A=\mathbf1(w\odot r)^T,\qquad
 \rho(A)=(w\odot r)^T\mathbf1
       =\sum_j\frac{n_j}{N}\frac{B_j}{n_j}
       =\frac MN=K.
\]
\begin{block}{Exact result}
Every fixed-count assignment gives $\Lambda=K$. Therefore
$\mathbb E_0[\Lambda]=K$ and $\operatorname{Var}_0(\Lambda)=0$.
\end{block}
\textbf{Scope:} complete connectivity alone is insufficient; the affinities must give uniform mixing over people. This uses the original model's self-draw convention.
\medskip

\footnotesize If self-draws are excluded at the individual level, the complete equal-weight graph instead gives $(M-1)/(N-1)$ for $M\geq1$ (and zero when $M=0$).
''')
frame('Two toy social graphs',r'''
Four neighborhoods, five residents each: $N=20$. Each matrix entry is $f(i\to j)$; diagonal entries include co-location.
\begin{columns}[T]
\begin{column}{.48\textwidth}
\textbf{Graph A: complete mixing}
\[
 F_A=\begin{pmatrix}1&1&1&1\\1&1&1&1\\1&1&1&1\\1&1&1&1\end{pmatrix},\qquad P_A=F_A/4.
\]
Every resident is equally likely to be contacted.
\end{column}
\begin{column}{.48\textwidth}
\textbf{Graph B: two communities}
\[
 F_B=\begin{pmatrix}1&1&.02&.02\\1&1&.02&.02\\.02&.02&1&1\\.02&.02&1&1\end{pmatrix},\quad P_B=F_B/2.04.
\]
Interactions within $\{1,2\}$ or $\{3,4\}$ are 50 times stronger than across communities.
\end{column}
\end{columns}
\medskip
Both graphs are fully connected. Their different weights isolate the effect of social structure.
\begin{block}{Experiment}
For each $M=1,\ldots,19$, draw 10,000 independent assignments; compute $\rho(P\operatorname{diag}(B_i/5))$. Also enumerate all feasible count vectors with exact hypergeometric probabilities.
\end{block}
''')
a=toy[toy.graph=='fully_mixed'];b=toy[toy.graph=='two_communities']
body=line([0,1],[0,1],'black,dashed','Reference $K$')+line(a.K,a['mean'],'MyNavy,only marks,mark=o','Graph A: simulation')+line(b.K,b.exact,'MyTeal,thick','Graph B: exact mean')+line(b.K,b['mean'],'MyTeal,only marks,mark=*','Graph B: simulation')
frame(r'Toy graphs: $K$ versus the random-assignment mean',axis(body,'Group fraction $K$','Mean PSSI', 'xmin=0,xmax=1,ymin=0,ymax=1')+r'\footnotesize Points: 10,000 assignments at each $K$. Solid curve: exact enumeration. At $K=0.5$, Graph B has exact mean $0.578069$, versus $0.500000$ for Graph A.')
body=line([.05,.95],[0,0],'black,dashed','Zero bias')+line(b.K,(b.exact-b.K)/(1-b.K),'MyTeal,thick','Graph B: exact bias')
body+=r'\addplot[MyNavy,only marks,mark=*,error bars/.cd,y dir=both,y explicit] coordinates {'+' '.join(f'({r.K:.6g},{r.adjusted_bias:.9g}) +- (0,{1.96*r.se/(1-r.K):.9g})' for r in b.itertuples())+r'};\addlegendentry{Simulation with 95\% MC CI}'
frame(r'Toy graphs: the expected bias after $K$ normalization',axis(body,'Group fraction $K$',r'$(m_0-K)/(1-K)$','xmin=0,xmax=1,legend pos=north west')+r'\footnotesize Graph A has zero bias at every $K$. Graph B has bias $0.15614$ at $K=0.5$. Error bars quantify uncertainty in the estimated mean, not the spread of individual assignments.')
frame('Largest saved PSSI matrix: Los Angeles',r'''
\begin{itemize}
\item $1{,}114\times1{,}114$ population-weighted transition matrix; $N=3{,}898{,}747$ residents. Largest among the 11 primary multicity PSSI matrices (NYC's PSSI matrix has 70 sections).
\item Use the saved primary walking/transit cost model, including its documented distance-only completion for missing travel pairs. These are modeled affinities, not observed contacts.
\item Hold this $P$ and all $n_i$ fixed. At each target $K\in\{.1,.3,.5,.7,.9\}$, assign $M=\operatorname{round}(KN)$ labels and use the actual fraction $M/N$.
\item 400 independent allocations per $K$; same raw PSSI definition as in the deck. Seed: 20261024.
\end{itemize}
\medskip
\textbf{Scope of ``largest'':} largest saved PSSI matrix; the repository also contains larger sparse geographic and Fryer graphs with different statistics.
\medskip
\footnotesize Source: \texttt{outputs/multicity/los\_angeles/analysis/analysis\_matrices.npz}. The graph is fixed here; only resident labels are randomized.
''')
frame('An exponential random-matrix model: explicit assumptions',r'''
\textbf{Alternative ensemble:} draw $F_{ij}\overset{\mathrm{iid}}\sim\operatorname{Exp}(1)$ for all ordered pairs, independently of labels. Keep the actual $n_j$, and set
\[
 P^{(F)}_{ij}=\frac{F_{ij}n_j}{\sum_\ell F_{i\ell}n_\ell}.
\]
\begin{itemize}
\item Directed independent edges; no geography, community effects, reciprocity or special diagonal. All moments are finite; every edge is positive.
\item The exponential rate cancels under row normalization. This is an assumption about raw $f$, not about normalized probabilities $P_{ij}$.
\item For equal $n_j$, rows are Dirichlet$(1,\ldots,1)$; the scaled eigenvalue bulk follows the random-Markov circular law asymptotically.
\item Unequal populations break this exact Dirichlet model. The Perron root of $P$ is always 1; the bulk law alone does not determine $\rho(P\operatorname{diag}(r))$.
\end{itemize}
\footnotesize Bordenave, Caputo \& Chafa\"i (2012), \href{https://arxiv.org/abs/0808.1502}{Circular Law Theorem for Random Markov Matrices}. We use finite-size simulation for the unequal-population ensemble.
''')
frame('Exponential weights are a benchmark, not an empirical fit',r'''
Saved Los Angeles off-diagonal affinities, divided by their mean:
\begin{center}\begin{tabular}{lrr}
 & Saved affinities & Exponential(1)\\\hline
Coefficient of variation & 1.165 & 1.000\\
10th percentile & 0.021 & 0.105\\
Median & 0.613 & 0.693\\
90th percentile & 2.376 & 2.303\\
99th percentile & 5.762 & 4.605\\
\end{tabular}\end{center}
\medskip
Reciprocity correlation $\operatorname{Corr}(F_{ij},F_{ji})=0.996$ over ordered entries, inconsistent with independent directed affinities.
\medskip
The saved model uses $f=\exp(-\beta C)$ with bounded, geography-dependent affinities. Exponentiating a cost is not the same as sampling an exponential random variable.
\begin{block}{Implication}
An iid exponential ensemble can illustrate dense random mixing, but its distribution must not be substituted for the fixed Los Angeles graph's null distribution.
\end{block}
''')
frame('Random-matrix perturbation on the actual graph',r'''
Write $r=K\mathbf1+\varepsilon$, so the random matrix is
$A=KP+P\operatorname{diag}(\varepsilon)$. Let
\[
 \pi^TP=\pi^T,\quad\pi^T\mathbf1=1,\qquad
 \Pi=\mathbf1\pi^T,\quad
 H=(I-P+\Pi)^{-1}(P-\Pi).
\]
Perturbing the simple Perron eigenvalue of $KP$ gives
\[
 \boxed{\Lambda\approx K+\pi^T\varepsilon+
 \frac1K\pi^T\operatorname{diag}(\varepsilon)H\varepsilon.}
\]
\textbf{Assumptions:} $K>0$; irreducible $P$; perturbations sufficiently small relative to the reduced resolvent scale. A simple eigenvalue is required, and non-normality can amplify perturbations.
\medskip
A Gaussian description additionally requires a finite-population CLT for the weighted counts (no dominant resident contribution). Tiny neighborhoods can violate a uniform small-noise assumption; numerical validation is essential.
''')
frame('Mean and variance from the assignment covariance',r'''
The fixed-count multivariate hypergeometric model gives exactly
\[
 C=\operatorname{Cov}(\varepsilon)
 =\frac{K(1-K)}{N-1}\left[N\operatorname{diag}(1/n_i)-\mathbf1\mathbf1^T\right].
\]
Taking expectations in the expansion,
\[
 \boxed{m_0\approx K+\frac1K\sum_{i,j}\pi_iH_{ij}C_{ij}},\qquad
 \boxed{\operatorname{Var}(\Lambda)\approx\pi^TC\pi.}
\]
The variance formula is first order; the mean correction is second order.
\medskip
If the linear term dominates and its CLT applies,
\[
 \Lambda\ \dot\sim\ \mathcal N\!\left(K+\frac1K\sum_{i,j}\pi_iH_{ij}C_{ij},\;\pi^TC\pi\right).
\]
\footnotesize The normal law is an approximation, not an exact finite-sample distribution. If $\pi=n/N$, the linear variance vanishes under fixed totals; quadratic terms then matter. For complete mixing $H=0$, recovering the exact point mass at $K$.
''')
frame('Los Angeles: predicted mean and fluctuation scale',r'''
The saved graph gives $|\lambda_2(P)|=0.6304$. Substituting its stationary weights and resolvent into the formulas yields
\[
 \boxed{m_0\approx K+(1-K)\,2.9410\times10^{-6}},
\]
\[
 \boxed{\operatorname{SD}(\Lambda)\approx
 \sqrt{K(1-K)\,4.2756\times10^{-8}}.}
\]
At $K\approx0.5$: predicted mean $0.50000160$, predicted SD $0.00010339$.
\medskip
The predicted expected normalized bias is $2.9410\times10^{-6}$ throughout this interior-$K$ approximation.
\begin{block}{Numerical check}
Across the five $K$ values, the second-order formula reproduces individual simulated Perron roots with RMSE between $1.2\times10^{-10}$ and $8.6\times10^{-9}$.
\end{block}
\footnotesize This supports the expansion for these tested allocations. It does not establish a uniform error bound over every allocation or near $K=0$.
''')
body=line(large.K,1e6*(large.predicted_mean-large.K),'MyTeal,thick,mark=square*','Second-order prediction')+line([.1,.9],[0,0],'black,dashed','Reference $K$')
body+=r'\addplot[MyNavy,only marks,mark=*,error bars/.cd,y dir=both,y explicit] coordinates {'+' '.join(f'({r.K:.9g},{1e6*(r.mean-r.K):.9g}) +- (0,{1.96e6*r.se:.9g})' for r in large.itertuples())+r'};\addlegendentry{400 assignments: 95\% MC CI}'
frame(r'Los Angeles: deviations of the mean from $K$',axis(body,'Group fraction $K$',r'$10^6(\widehat m_0-K)$','xmin=.05,xmax=.95,legend pos=south west')+r'\footnotesize A residual plot resolves differences hidden on a 0--1 scale. Every interval contains $K$ and the second-order prediction; 400 draws do not resolve a bias of only a few parts per million.')
raw=pd.read_csv(O/'large_draws.csv');r=large.iloc[2];vals=raw[np.isclose(raw.K,.5)]['value'].to_numpy();u=(vals-r.predicted_mean)/r.predicted_sd
hist,edges=np.histogram(u,bins=np.linspace(-3.5,3.5,18),density=True);xx=np.linspace(-3.5,3.5,101)
body=line((edges[:-1]+edges[1:])/2,hist,'MyNavy,ybar,bar width=6pt,fill=MyNavy!35','400 assignments')+line(xx,norm.pdf(xx),'MyTeal,thick,no marks','Predicted normal density')
frame('Los Angeles: distribution at a 50\% group share',axis(body,'$(\Lambda-m_{\mathrm{pred}})/s_{\mathrm{pred}}$','Density','xmin=-3.5,xmax=3.5,legend pos=north east')+r'\footnotesize Observed SD $0.00010448$ versus predicted $0.00010339$. Empirical central 95\% interval: $[0.499790,\,0.500209]$. This interval describes assignment variability, not uncertainty in the mean.')
frame('Exponential ensemble versus the actual graph',r'''
At target $K=0.5$, retain the same 1,114 neighborhood populations.
For the exponential ensemble, independently redraw both the entire affinity matrix and the assignment in each of 300 replicates.
\begin{center}\begin{tabular}{lcc}
 & Actual fixed graph & iid exponential graph\\\hline
Replicates & 400 & 300\\
Mean PSSI & 0.50000912 & 0.50000046\\
Monte Carlo SE & 0.00000522 & 0.00000048\\
Assignment/ensemble SD & 0.00010448 & 0.00000832\\
\end{tabular}\end{center}
\begin{block}{Finding}
Both means are close to $K$, but the exponential ensemble has about 12.6 times less standard deviation. It misses substantial fluctuation structure of the actual graph.
\end{block}
\footnotesize Actual $K=1{,}949{,}374/3{,}898{,}747=0.5000001282$. The exponential column averages over graph uncertainty as well as resident assignments; it is a different null model.
''')
frame(r'When is the $K$ approximation adequate?',r'''
\begin{itemize}
\item \textbf{Equal-strength complete mixing:} $m_0=K$ exactly, including unequal neighborhood populations under the stated self-draw convention.
\item \textbf{Toy communities:} exact $m_0=0.578069$ at $K=0.5$. Connectivity does not ensure exact centering; expected normalized bias is $0.15614$.
\item \textbf{Saved Los Angeles PSSI:} $K$ is practically accurate under uniform resident reassignment at the five tested shares. All 90\% Monte Carlo CIs for normalized bias lie inside the illustrative tolerance $[-0.02,0.02]$.
\item \textbf{Distribution:} graph-specific perturbation predicts the observed scale well. An iid exponential benchmark is substantially too concentrated and does not match the observed dependencies.
\end{itemize}
\medskip
The conclusion concerns this fixed graph and this resident-level null. Permuting neighborhood shares, assigning one label per neighborhood, or using component-averaged SSI would define different experiments.
''')
frame('Derivation of the second-order correction',r'''
Put $D=\operatorname{diag}(\varepsilon)$ and consider $A(t)=KP+tPD$.
Write $v(t)=\mathbf1+t u+O(t^2)$ with $\pi^Tu=0$, and
$\lambda(t)=K+t a+t^2 b+O(t^3)$.
\[
 K(I-P)u=P\varepsilon-a\mathbf1,\qquad a=\pi^T\varepsilon.
\]
The normalized solution is $u=H\varepsilon/K$. At second order, multiply the eigenvector equation by $\pi^T$:
\[
 b=\pi^TPD u=\pi^TDu
   =\frac1K\pi^T\operatorname{diag}(\varepsilon)H\varepsilon.
\]
Since $\mathbb E\varepsilon=0$ and $\mathbb E[\varepsilon_i\varepsilon_j]=C_{ij}$,
\[
 \mathbb E b=\frac1K\sum_{i,j}\pi_iH_{ij}C_{ij}.
\]
\footnotesize This is a local Taylor expansion about $t=0$ evaluated at $t=1$; finite-sample accuracy is checked against the computed eigenvalues rather than assumed from the matrix dimension.
''')
frame('Reproducibility and sources',r'''
\textbf{Simulation:} \texttt{scripts/analyze\_k\_approximation.py}. Seed 20261024. Hypergeometric sampling preserves every neighborhood's population and the total number of group labels. Toy means are checked against exact count-vector enumeration.
\medskip

\textbf{Numerics:} dense eigenvalues for the toy matrices; ARPACK dominant eigenpairs for the large matrices, with residual checks below $10^{-8}$. Theory uses the saved $P$, its stationary vector and reduced resolvent.
\medskip

\textbf{Outputs:} \texttt{outputs/k\_approximation/} contains the mean/SD/interval tables, individual large-matrix draws, matrix inventory, source hash and assumptions in the accompanying README. Plots are editable PGFPlots coordinates embedded in the source.
\medskip

\textbf{Random-matrix reference:} Bordenave, Caputo and Chafa\"i, \emph{Circular Law Theorem for Random Markov Matrices}, PTRF 152 (2012), 751--779. \href{https://arxiv.org/abs/0808.1502}{arXiv:0808.1502}. The circular law concerns the bulk spectrum; the PSSI mean formulas here are the displayed perturbation derivation.
''')
content='\n'.join(frames)
original=(ROOT/'segregation_slides_formatted.tex').read_text();preamble=original.split(r'\begin{document}')[0]
(S/'k_approximation_frames.tex').write_text(content)
(S/'k_approximation.tex').write_text(preamble+r'\begin{document}'+'\n'+content+r'\end{document}'+'\n')
start=original.index(r'\begin{frame}[shrink=0]{Further analysis: graph structure changes the null mean}')
end=original.index(r'\begin{frame}[shrink=0]{A model where normalization recovers segregation exactly}',start)
updated=original[:start]+content+'\n'+original[end:]
updated=updated.replace(r'\includegraphics[width=0.18\textwidth,height=1.45cm,keepaspectratio]{Columbia shield.png}',r'\IfFileExists{Columbia shield.png}{\includegraphics[width=0.18\textwidth,height=1.45cm,keepaspectratio]{Columbia shield.png}}{\vspace{1.45cm}}')
(ROOT/'segregation_slides_k_analysis.tex').write_text(updated)
print(f'{len(frames)} replacement slides; standalone and full deck written.')
