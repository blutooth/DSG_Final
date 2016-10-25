from zipfile import ZipFile
import pandas as pd

fname2='./submissions/Y_test_delta_1.predict'
fname1='./submissions/Y_test_delta_1.predict'
Test1=pd.read_csv(fname1)
Test2=pd.read_csv(fname2)
combined=(Test1+Test2)/2
fout=open("./submissions/Y_out_test",'w+')
for pred in combined.iloc[:,0]:
            fout.write(str(pred) + '\n')
