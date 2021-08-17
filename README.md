# number_creator_GAN
creates photos of numbers between 0 and 9 using ANN

## requirement
1. tensorflow 2.41 and above
2. numpy
## usage
```python
from num_GAN import generate_numbers
img=generate_numbers()[x]
```
where x is number between 0 and 9 that you want the image to contain. 
for example if you want image of the number 5 
```python
from num_GAN import generate_numbers
img=generate_numbers()[5]
```