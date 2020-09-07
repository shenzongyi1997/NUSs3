**Shape:**$m$ is the number of samples, $n$ is the number of dimension.

![1599386550872](assets/1599386550872.png)

Compute Gradient:

**Linear regression:** $\frac{\partial L}{\partial {\bf w}} = (\hat y - y){\bf x}$ (squared error)

**Logistic regression**: $\frac{\part L}{\part {\bf w}} = (p-y) {\bf x}$ (cross-entropy loss) SE will "gradient vanishing"

**Softmax regression:** $\frac{\part L}{\part W} = ({\bf p - y}){\bf x^T}$

