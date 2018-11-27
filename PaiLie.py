from itertools import combinations

n = 7
sum = 0
if n % 2 == 0:
    km = n/2
else:
    km = (n+1)/2
km = int(km)
for k in range(km+1):
    combins = [c for c in combinations(range(n-k+1), k)]
    a = len(combins)
    sum += a
print(sum)