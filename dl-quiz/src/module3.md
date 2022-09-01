[![Dataflowr](https://raw.githubusercontent.com/dataflowr/website/master/_assets/dataflowr_logo.png)](https://dataflowr.github.io/website/)

# Module 3: Loss functions for classification

These are the quizzes corresponding to [Module 3](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/)

In all the questions below we assume that all the import have been done 
```python
import torch
import torch.nn as nn
```
Try to answer the questions without running the code ;-)

We run the following code: 

```python
> m = nn.Sigmoid() 
> loss = nn.BCELoss() 
> target = torch.empty(3).random_(2) 
> print(target) 
tensor([0., 1., 1.])
> input = torch.randn(3, requires_grad=True) 
> optimizer = torch.optim.SGD([input], lr = 0.1)
> print(m(input)) 
tensor([0.3517, 0.4834, 0.3328], grad_fn=<SigmoidBackward>)
> for _ in range(10000): 
    output = loss(m(input), target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
> print(xxxxxx)
```
and obtain the following result: 
```python
tensor([0.0030, 0.9970, 0.9970], grad_fn=<SigmoidBackward>)
```

{{#quiz quiz_31.toml}}

We run the following code: 

```python
> target = torch.empty(1,2,3).random_(2) 
> print(target)
tensor([[[1., 1., 1.],
    [0., 0., 1.]]])
> input = torch.randn((1,2,3), requires_grad=True)
> optimizer = torch.optim.SGD([input], lr = 0.1)
> print(m(input).size())
torch.Size([1, 2, 3])
```
We then run 
```python
for _ in range(10000):
    output = loss(m(input), target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
```

{{#quiz quiz_32.toml}}

We run the following code:
```python
> target = torch.randn(3)
> print(target)
tensor([-0.1272, -0.4165,  0.1002])
> input = torch.randn(3, requires_grad=True)
> optimizer = torch.optim.SGD([input], lr = 0.1)
> print(m(input))
tensor([0.5203, 0.6769, 0.6586], grad_fn=<SigmoidBackward>)
```
and then
```python
for _ in range(10000):
    output = loss(m(input), target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(m(input))
```
{{#quiz quiz_33.toml}}

We run the following code:
```python
> target = 10*torch.randn(3)
> print(target)
tensor([ 12.1225, -11.8731,  19.2255])
> input = torch.randn(3, requires_grad=True)
> optimizer = torch.optim.SGD([input], lr = 0.1)
> print(m(input))
tensor([0.2664, 0.7103, 0.5226], grad_fn=<SigmoidBackward>)
```
and then
```python
for _ in range(10000):
    output = loss(m(input), target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(m(input))
```
{{#quiz quiz_34.toml}}

We run:
```python
> loss2 = nn.BCEWithLogitsLoss()
> target = torch.empty(3).random_(2)
> print(target)
tensor([0., 1., 0.])
```
and then code1:
```python
input = torch.randn(3, requires_grad=True)
optimizer = torch.optim.SGD([input], lr = 0.1)
for _ in range(500):
    output = loss2(m(input), target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(m(input))
```
and then code2:
```python
input = torch.randn(3, requires_grad=True)
optimizer = torch.optim.SGD([input], lr = 0.1)
for _ in range(500):
    output = loss2(input, target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(m(input))
```

{{#quiz quiz_35.toml}}

We run:
```
> loss3 = nn.NLLLoss()
> m3 = nn.LogSoftmax(dim=1)
> target = torch.empty(4).random_(6)
> print(target)
tensor([1., 5., 0., 2.])
```
then code1:
```
input = torch.randn(4, requires_grad=True)
optimizer = torch.optim.SGD([input], lr = 0.1)
for _ in range(1000):
    output = loss3(m3(input), target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(torch.exp(m3(input)))
```
code2:
```
input = torch.randn((4,7), requires_grad=True)
optimizer = torch.optim.SGD([input], lr = 0.1)
for _ in range(1000):
    output = loss3(m3(input), target.long())
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(torch.exp(m3(input)))
```
code3:
```
input = torch.randn((4,6), requires_grad=True)
optimizer = torch.optim.SGD([input], lr = 0.1)
for _ in range(1000):
    output = loss3(m3(input), target)
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(torch.exp(m3(input)))
```
{{#quiz quiz_36.toml}}

We now take
```python
> target = torch.empty(4,2,3).random_(6)
> print(target)
tensor([[[3., 1., 0.],
         [4., 2., 3.]],

        [[1., 3., 3.],
         [0., 1., 0.]],

        [[2., 1., 1.],
         [0., 4., 5.]],

        [[1., 3., 5.],
         [3., 2., 5.]]])
```
{{#quiz quiz_37.toml}}

We now run
```python
loss4 = nn.CrossEntropyLoss()
m4 = nn.Softmax(dim=1)
target = torch.empty(4).random_(6)
input = torch.randn((4,6), requires_grad=True)
optimizer = torch.optim.SGD([input], lr = 0.1)
```
then code1
```python
for _ in range(500):
    output = loss4(input, target.long())
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(m4(input))
```
code2
```python
for _ in range(500):
    output = loss3(input, target.long())
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(m4(input))
```
code3
```python
for _ in range(500):
    output = loss3(m3(input), target.long())
    optimizer.zero_grad()
    output.backward()
    optimizer.step()
print(torch.exp(m3(input)))
```
{{#quiz quiz_38.toml}}


If you did not make any mistake, you can safely go to [Module 4](https://dataflowr.github.io/website/modules/4-optimization-for-deep-learning/)