import itertools
import subprocess

from tqdm import tqdm
learning_rate_list = [1e-4,1e-5]
regularize_list = [1e-3,1e-4,1e-5]
alpha_list = [1e-3,1e-4,1e-5]
listoflist = []
listoflist.append(learning_rate_list)
listoflist.append(regularize_list)
listoflist.append(alpha_list)

iterations= list(itertools.product(*listoflist))[2:]
for l, r, a in tqdm(iterations):
    subprocess.run("python main.py --learning_rate "+l.__str__()+" --alpha "+a.__str__()+" --regularization_parameter "+r.__str__())#+" --dropout "+d.__str__())

