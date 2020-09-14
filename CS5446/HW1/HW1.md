This is for CS5446 Homework1 written assignment by A0196990J, e0392432@u.nus.edu.

Thanks for reading!

### Q1:Classcial Planning

1. (a)what (i) ignores is a proper subset of what (ii) ignores, so **(ii) is more relaxed**, and this means **heuristic (i) dominants (ii)**.  And Both (i) and (ii) ignores part of the precondition, so they are both **admissible heuristics**. Finally, **(i)** will lead to fewer nodes explored.


- (b)This sentence "each plane can only carry one cargo" has 2 possible explanations to me, I will answer them separately:

  - A plane can carry one cargo **during its lifetime**, for this case, the following will be done:
    - Introduce a  fluent called $Loaded(p)$.  
    - In $Action(Load(c,p,a))$, add $\neg Loaded(p)$ to its **precondition**, and add $Loaded(p)$ to its **effect**.
  - A plane can carry one cargo **during one flying process**, for this case, the following will be done:
    - Introduce a  fluent called $Loaded(p)$.  
    - In $Action(Load(c,p,a))$, add $\neg Loaded(p)$ to its **precondition**, and add $Loaded(p)$ to its **effect**.
    - In $Action(Unload(c,p,a))$, add $Loaded(p)$ to its **precondition**, and add $\neg Loaded(p)$ to its **effect**.

- Based on the following equation:

  ![1598881595457](assets/1598881595457.png)

  We can get that the answer is:
  $$
  At(P_1,SFO)^{t+1}\Leftrightarrow Fly(P1,JFK,SFO)^t\or(At(P_1,SFO)^t\and\neg Fly(P_1,SFO,JFK)^t)
  $$

### Q2: Decision Theory

- (a): Because Bob is rational, he is likely to seek the max expectation of utility. Though $C$ ensures 40 utility, the lottery will give a $100*0.6+0=60$ utility, Bob will choose the lottery.

- (b) Based on the previous question, let's imagine this scenario: Alice can choose from 

  - (1) a lottery with {$p,U(x_1);1-p,U(x_2)$} where $x_1<x_2$. To see a concrete example, we can set $p=\frac{x_2}{x_1+x_2}$, so $1-p={x_1}{x_1+x_2}$.
  - (2) $U(x_3)$ where $x_1<x_3<x_2$, and $x_3=px_1+(1-p)x_2$. In the example, it's $\frac{2x_1x_2}{x_1+x_2}$.

  Because her utility $U(x)=x^2$, so based on Jensen's inequality, $E(U(x))>U(E(x))$, that means, the expectation of the lottery $pU(x_1)+(1-p)U(x_2)$ is always larger than $U(x_3)$. If we use the concrete example, the expectation of the lottery is 
  $$
  E=pU(x_1)+(1-p)U(x_2) = \frac{x_1^2x_2+x_1x_2^2}{x_1+x_2}=x_1x_2
  $$
  and the $U(x_3)=\frac{4x_1^2x_2^2}{(x_1+x_2)^2}$

  So $E/U(x_3)$ = $\frac{(x_1+x_2)^2}{4x_1x_2}>1$

  So we can see, Alice will always prefer a lottery. So she is risk-seeking.

- (c) This is quite straightforward, if Cathy prefers $C $ to $D$, this means her utility function $U$ satisfies $U(C)>U(D)$, which means $0.2U(A)+0.8U(B)>0.3U(A)+0.7U(D)$. This will lead to $U(A)<U(B)$. This is contradictory to the claim "Cathy prefers $A$ to $B$". So $U(A)>U(B)$ and at the same time $U(A)<U(B)$, there is not a such thing.