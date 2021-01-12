# coding=UTF-8
from os import listdir, walk
from os.path import isfile, isdir, join, abspath
import pandas as pd
import numpy as np
import random
import argparse
import re
import xlrd

parser = argparse.ArgumentParser()
parser.add_argument("--root", dest="root", help="Given dataset path", default="./")
parser.add_argument("-o","--output", dest="output", help="Given output path", default="./")
args = parser.parse_args()
print(args.root)

root=args.root


l=[]
ans=[]
for i in listdir(root):
    if i.find(".")<0 and isdir(join(root,i)):
        l.append(join(root,i))
'''for i in range(len(l)):
    for root, dirs, files in walk(l[i]):
        if len(dirs)>0:
            l[i]=join(l[i],dirs[0])
            break
    print(l[i])'''
for i in range(len(l)):
    #print(listdir(l[i]))
    # ans text
    print(l[i])
    for k in listdir(l[i]):
        if k.find(".xlsx")>=0 and k.find("決賽簡答題25題")<0:
            ans.append(join(l[i],k))
    for k in listdir(l[i]):
        # data or wav
        if k.find("data") >=0 or k.find("wav")>=0:
            l[i]=join(l[i],k)

#train/dev/test
wav=[]
text=[]
for i in range(len(l)):
    wb = xlrd.open_workbook(ans[i], encoding_override='utf-8')
    df = pd.read_excel(wb)
    # df = pd.read_csv(ans[i])
    # df = pd.read_excel(ans[i], encoding="utf-8")
    txt=df.columns.values[1]
    idx=list(df[df.columns.values[0]])
    #
    if l[i].find("決賽簡答題25題")>=0:
        continue
    # wav
    for root, dirs, files in walk(l[i]):
        #trainfile.append(j)
        if len(files)>0 and files[0].find(".wav")>0:
            for j in files:
                #print(j)
                #print(int(j[1:j.find(".wav")]))
                
                try:
                    idx.index(int(j[1:j.find(".wav")]))+1
                    #print(idx.index(int(j[1:j.find(".wav")]))+1)
                    wav.append(join(root,j))
                except:
                    #print("Cannot find :"+str(int(j[1:j.find(".wav")]))+ " in excel label")
                    continue
                if j[0]=="A":
                    text.append(df.iat[idx.index(int(j[1:j.find(".wav")])),1])
                elif j[0]=="B":
                    text.append(df.iat[idx.index(int(j[1:j.find(".wav")])),2])
                elif j[0]=="C" or j[0]=="c":
                    text.append("一"+str(df.iat[idx.index(int(j[1:j.find(".wav")])),3])+
                               "二"+str(df.iat[idx.index(int(j[1:j.find(".wav")])),4])+
                               "三"+str(df.iat[idx.index(int(j[1:j.find(".wav")])),5])+
                               "四"+str(df.iat[idx.index(int(j[1:j.find(".wav")])),6]))
                else:
                    print("EXCEPT: "+ j)



##
#過濾標點符號
i=0
while i <(len(wav)):
    #先代換掉\n\t
    #有ASC不可視字元的應該是有問題，直接刪掉
    text[i]=re.sub(r"[\n\t]","", text[i])
    t=re.search(r"([\u0000-\u001F\u007F\u0080-\u009F])", text[i])
    
    
    
    if t!=None:
        #print(t)
        #print(text[i])
        #REMOVE
        del text[i]
        del wav[i]
        continue
    else:
        #修正 'ㄧ' ->'一'
        text[i]=re.sub(r"[ㄧ]","", text[i])
        # remove 標點符號
        text[i]=re.sub(r"[’!#$%&\\()｢*+,-./:;<=>?@，﹐。◎﹑—?★、…【】《》？“”‘’！~^「」『』：∶；﹔⋯・•·‧¨─¬ˋ`⑵⑶⑷⑤⑹⑺⑾⑼③④⑸⑻﹖﹗ 　\"\'\[\]]","", text[i])
        # remove (...)
        text[i]=re.sub(r"（.*）","", text[i])
        #標點符號
        text[i]=re.sub(r"[\u3000\u3001-\u303F]","", text[i])
        
    #https://blog.miniasp.com/post/2019/01/02/Common-Regex-patterns-for-Unicode-characters
    # ○(零):\u25CB  ℃:\u2103 °:\u00B0
    t=re.search(r"[^\u0041-\u005A\u0061-\u007A\u0020\u3000\u0030-\u0039\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF\uFF01-\uFF5E\u3105-\u3129\u25CB\u2103\u00B0]", text[i])
    if t!=None:
        print(t)
        #print(text[i])
        #REMOVE
        del text[i]
        del wav[i]
        continue
    else:
        # replace
        i=i+1
        

## shuffle
'''indices = np.arange(len(wav))
wav=np.array(wav)
text=np.array(text)
np.random.shuffle(indices)
wav = wav[indices]
text = text[indices]'''

list_zipped = list(sorted(zip(wav,text)))
random.seed(10)
random.shuffle(list_zipped)
wav,text = zip(*list_zipped) #unzipping

## output
fixedlen=len(wav)//10
num=0
for c in ["train","dev","test"]:
    root=args.output
    wavscp_text=""
    text_text=""
    utt2spk_text=""
    k=fixedlen
    if c=="train":
        k=len(wav)-2*fixedlen
    for i in range(k):
        #print(num)
        #print(text[num])
        wavscp_text+=c+"_"+str(i).zfill(5)+" "+join(root,wav[num])+"\n"
        text_text+=c+"_"+str(i).zfill(5)+" "+text[num]+"\n"
        utt2spk_text+=c+"_"+str(i).zfill(5)+" "+c+"_"+str(i).zfill(5)+"\n"
        num=num+1
    with open(join(root,c+"_wav.scp"),"w",encoding="utf-8") as f:
        f.write(wavscp_text)
    with open(join(root,c+"_text"),"w",encoding="utf-8") as f:
        f.write(text_text)
    with open(join(root,c+"_utt2spk"),"w",encoding="utf-8") as f:
        f.write(utt2spk_text)
        