import scipy.io as scio

print(scio.loadmat('signed_weight_before.mat')['w'][0])
print(scio.loadmat('signed_weight_after.mat')['w'][0])

print(scio.loadmat('none_weight_before.mat')['w'][0])
print(scio.loadmat('none_weight_after.mat')['w'][0])
