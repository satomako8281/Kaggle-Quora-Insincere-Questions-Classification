import pandas as pd


sbase = "".join(chr(i) for i in range(0,65535) if chr(i).isalnum())

def encode(bin_str, sbase):
    n = int(bin_str,2)
    code = []
    base = len(sbase)
    while n > 0:
        n, c = divmod(n, base)
        code.append(sbase[c])
    code = reversed(code)
    return ''.join(code)

def decode(code, sbase, size):
    code = reversed(code)
    base = len(sbase)
    bin_str =  bin(sum([sbase.index(c)*base**i for i, c in enumerate(code)]))[2:]
    bin_str = (size-len(bin_str))*'0' + bin_str
    return bin_str


df = pd.read_csv('~/tmp/submission9_lasso.csv')
mojiretu = ''
for x in df['prediction'].values.tolist():
    mojiretu += str(x)
print(mojiretu)

