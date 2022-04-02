import json
import cv2
import pickle
import os
import numpy as np
import math
from scipy.cluster.vq import vq, kmeans, whiten
import pickle
import sys

root = "C:\\Users\\zxk\\Desktop\\capstone\\prime-joints\\prime-joints\\"
operator_path = "..\\operator\\"
center_path = "..\\center\\"
index_path = "..\\index\\"
histogram_path = "..\\histogram\\"
train_bound = 74
cluster_num = 256
sup = [1,1,1,2,3,1,5,6,1,8 ,9 ,1 ,11,12,0 ,14,0 ,16]
sub = [0,1,3,4,4,6,7,7,9,10,10,12,13,13,15,15,17,17]
alp_limbs = [[1,0,2],[1,2,5],[1,2,8],[1,8,11],[1,5,11],[2,1,3],[3,2,4],[5,1,6],[6,5,7],[8,1,9],[9,8,10],[11,1,12],[12,11,13]]
views = ['000','018','036','054','072','090','108','126','144','162','180']
gallery_list = ["nm-01","nm-02","nm-03","nm-04"]
fea_kinds = "PASTV"
fea_length = [6,6,6,8,2]

def computePLP(s_p):
    #computes the distance and angle to center and sup/sub point
    feature=[]
    x=s_p[1*3]
    y=s_p[1*3+1]
    H=(s_p[8*3+1]+s_p[9*3+1])/2-s_p[1*3+1]+1e-3 #optimize point
    #print(H)
    if H<10 or H>100:
        return []
    for i in range(18):
        #center
        x1=s_p[i*3]-x
        y1=s_p[i*3+1]-y
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)
        #sup
        x1=s_p[i*3]-s_p[sup[i]*3]
        y1=s_p[i*3+1]-s_p[sup[i]*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)
        #sub
        x1=s_p[i*3]-s_p[sub[i]*3]
        y1=s_p[i*3+1]-s_p[sub[i]*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)
    return feature

def computeALP(pre_skdts,next_p):
    pres_p,s_p = pre_skdts[0],pre_skdts[1]
    tempofeature=[]
    H=(s_p[8*3+1]+s_p[9*3+1])/2-s_p[1*3+1]+1e-3
    for i in range(18):
        x1=pres_p[i*3]-s_p[i*3]
        y1=pres_p[i*3+1]-s_p[i*3+1]

        x2=next_p[i*3]-s_p[i*3]
        y2=next_p[i*3+1]-s_p[i*3+1]
        limb1=np.array([x1,y1])
        limb2=np.array([x2,y2])
        lx=np.sqrt(limb1.dot(limb1))/H
        ly=np.sqrt(limb2.dot(limb2))/H
        cos_angle=limb1.dot(limb2)/(lx*ly+0.001)
        cos_angle= cos_angle/(2*np.pi)+0.5
        tempofeature.append([lx,ly,cos_angle])

    feature=[]

    for alp_limb in alp_limbs:
        limb_a=np.array([s_p[alp_limb[2]*3]-s_p[alp_limb[0]*3],s_p[alp_limb[2]*3+1]-s_p[alp_limb[0]*3+1]])
        limb_b=np.array([s_p[alp_limb[1]*3]-s_p[alp_limb[0]*3],s_p[alp_limb[1]*3+1]-s_p[alp_limb[0]*3+1]])
        lx=np.sqrt(limb_a.dot(limb_a))/H
        ly=np.sqrt(limb_b.dot(limb_b))/H
        cos_angle=limb_b.dot(limb_a)/(lx*ly+0.001)
        if cos_angle<-1 or cos_angle>1:
            cos_angle=0
        cos_angle=np.arccos(cos_angle)/(2*np.pi)+0.5
        feature.append(lx)
        feature.append(ly)
        feature.append(cos_angle)
        feature.extend(tempofeature[alp_limb[0]])

    return feature

def computeSLP(pre_skdts,next_p):

    pres_p,s_p = pre_skdts[0],pre_skdts[1]
    feature=[]
    H=(s_p[8*3+1]+s_p[9*3+1])/2-s_p[1*3+1]+1e-3
    for i in range(18):
        x1=s_p[i*3]-s_p[sup[i]*3]
        y1=s_p[i*3+1]-s_p[sup[i]*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)
        x2=pres_p[i*3]-pres_p[sup[i]*3]
        y2=pres_p[i*3+1]-pres_p[sup[i]*3+1]
        dist=np.sqrt(pow(x1-x2,2)+pow(y1-y2,2))/H
        angle=math.atan2(y1-y2, x1-x2)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)
        x3=next_p[i*3]-next_p[sup[i]*3]
        y3=next_p[i*3+1]-next_p[sup[i]*3+1]
        dist=np.sqrt(pow(x1-x3,2)+pow(y1-y3,2))/H
        angle=math.atan2(y1-y3, x1-x3)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

    return feature

def computeTLP(pres_p4,pres_p2,pres_p1,s_p,next_p1,next_p2,next_p4):

    feature=[]
    H=(s_p[8*3+1]+s_p[9*3+1])/2-s_p[1*3+1]+1e-3
    for i in range(18):
        x1=pres_p1[i*3]-pres_p2[i*3]
        y1=pres_p1[i*3+1]-pres_p2[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

        x1=s_p[i*3]-pres_p1[i*3]
        y1=s_p[i*3+1]-pres_p1[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

        x1=next_p1[i*3]-s_p[i*3]
        y1=next_p1[i*3+1]-s_p[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

        x1=next_p2[i*3]-next_p1[i*3]
        y1=next_p2[i*3+1]-next_p1[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

        x1=pres_p4[i*3]-pres_p2[i*3]
        y1=pres_p4[i*3+1]-pres_p2[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

        x1=s_p[i*3]-pres_p2[i*3]
        y1=s_p[i*3+1]-pres_p2[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)


        x1=next_p2[i*3]-s_p[i*3]
        y1=next_p2[i*3+1]-s_p[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

        x1=next_p4[i*3]-next_p2[i*3]
        y1=next_p4[i*3+1]-next_p2[i*3+1]
        dist=np.sqrt(pow(x1,2)+pow(y1,2))/H
        angle=math.atan2(y1, x1)/(2*np.pi)+0.5
        feature.append(dist)
        feature.append(angle)

    return feature

def computeVLP(s_p,next_p):
    fea_frag = []
    for p in [s_p,next_p]:
        if p[3] == 0 or p[4] == 0 or p[24] == 0 or p[25] == 0 or p[33] == 0 or p[34] == 0:
            return fea_frag
    neck_list = [[s_p[3*1],s_p[3*1+1]],[next_p[3*1],next_p[3*1+1]]]
    hip_list = [[(s_p[3*8]+s_p[3*11])/2,(s_p[3*8+1]+s_p[3*11+1])/2],[(next_p[3*8]+next_p[3*11])/2,(next_p[3*8+1]+next_p[3*11+1])/2]]
    limb1 = np.array([neck_list[0][0]-hip_list[0][0],neck_list[0][1]-hip_list[0][1]])
    limb2 = np.array([neck_list[1][0]-hip_list[1][0],neck_list[1][1]-hip_list[1][1]])
    limb_a = limb2 - limb1
    limb_b = np.array([(neck_list[1][0]+hip_list[1][0]-neck_list[0][0]-hip_list[0][0])/2,(neck_list[1][1]+hip_list[1][1]-neck_list[0][1]-hip_list[0][1])/2])
    ly = np.sqrt(limb_b.dot(limb_b))
    lx = np.sqrt(limb_a.dot(limb_a))
    cos_angle=limb_b.dot(limb_a)/(lx*ly+0.001)
    if cos_angle<-1 or cos_angle>1:
        cos_angle=0
    cos_angle=np.arccos(cos_angle)/(2*np.pi)+0.5
    fea_frag = [round(x,4) for x in [lx/ly,cos_angle]]    
    return fea_frag

os.chdir(root)

# create folders
for _path in [operator_path,center_path,index_path,histogram_path]:
    if not os.path.exists(_path):
        os.mkdir(_path)

# extract operator features

ojs = os.listdir(root)
#paths = []
#count = 0
for oj in ojs:
    oj_p = os.path.join(root,oj)
    if os.path.isdir(oj_p):
        #paths.append(path)
        fl_ns = os.listdir(oj_p)
        fl_ns.sort()
        print(oj_p)
    
    if os.path.exists(operator_path+oj+".txt"):
        print("operator features already exist")
        continue

    skdts = [[],[],[],[],[],[],[],[]]
    feature_fragments = []

    ofile = open(operator_path+oj+".txt","w")
    ofile.write(str(len(fl_ns))+"\n")

    for fl_n in fl_ns:
        fl_p = os.path.join(oj_p,fl_n)
        if not os.path.exists(fl_p):
            print("frame ",fl_p," not found")
            continue
        with open(fl_p,"r") as fl:
            js_data = json.load(fl)
            if js_data["people"]==[]:
                print("frame data ",fl_p," not found")
                continue
            if js_data["people"][0]==[]:
                print("frame data ",fl_p," not found")
                continue
            skdt = js_data["people"][0]["pose_keypoints_2d"]
            skzero = np.array(skdt)
            skzero = np.where(skzero == 0)
            if len(skzero[0]) > 50:
                print ("frame data ",fl_p, "is zero")
                continue
        feature=computePLP(skdt)
        if feature==[]:
            print("invalide H")
            continue
        #print(len(feature))
        feature= [str(round(x,4)) for x in feature]
        str_convert = ','.join(feature)
        ofile.write("P,"+str_convert+"\n")

        if skdts[7]!=[]:
            feature_fragment = computeVLP(skdts[7],skdt)
            if feature_fragment == []:
                print("invalide V")
            else: 
                feature_fragments.extend(feature_fragment)
    
        if  skdts[6]!=[]:
            pre_skdts = skdts[6:]

            feature=computeALP(pre_skdts,skdt)#78
            #print(len(feature))
            feature= [str(round(x,4)) for x in feature]
            str_convert = ','.join(feature)
            ofile.write("A,"+str_convert+"\n")

            feature=computeSLP(pre_skdts,skdt)#18*6
            #print(len(feature))
            feature= [str(round(x,4)) for x in feature]
            str_convert = ','.join(feature)
            ofile.write("S,"+str_convert+"\n")

        if  skdts[0]!=[]:
            feature=computeTLP(skdts[0],skdts[2],skdts[3],skdts[4],skdts[5],skdts[6],skdt)#288
            #print(len(feature))
            feature= [str(round(x,4)) for x in feature]
            str_convert = ','.join(feature)
            ofile.write("T,"+str_convert+"\n")

        skdts.pop(0)
        skdts.append(skdt)

        # print(skdts)
        # if count >10:
        #     sys.exit(0)
        # count = count + 1

    str_convert = ','.join([str(x) for x in feature_fragments])
    ofile.write("V,"+str_convert+"\n")
    #print(oj,"  ",feature_fragments)

print("*****operator features extract successfully*****")

# create index file

train_index = index_path+"0_"+str(train_bound) +"_train_"
gallery_index = index_path+str(train_bound+1) +"_124_gallery_"
probe_index = index_path+str(train_bound+1) +"_124_probe_"

ojs = os.listdir(operator_path)
ojs.sort()

for view in views:
    with open(train_index+view+".txt","w") as ffile1,open(gallery_index+view+".txt","w") as ffile2,open(probe_index+view+".txt","w") as ffile3:
        for oj in ojs:
            oj_path = os.path.join(operator_path,oj)
            if oj.find('-'+view) >= 0:
                if int(oj[:oj.find("-")]) <= train_bound:
                    ffile1.write(oj_path+"\n")
                elif gallery_list.count(oj[4:9]) > 0:
                    ffile2.write(oj_path+"\n")
                else:
                    ffile3.write(oj_path+"\n")

print("*****index files create successfully*****")

# read operator features

op_features = [[],[],[],[],[]]
# op_fs = []
f_centers = []

for view in views:
    with open(train_index+view+".txt","r") as ffile:
        #index_lines = ffile.readlines()
        index_lines = ffile.readlines()[:100]
    
    for index_line in index_lines:
        index_line = index_line.replace("\n","")
        if os.path.exists(index_line):
            with open(index_line,"r") as ffile:
                feature_lines = ffile.readlines()[1:]
        else:
            print(index_line," file not found")
            continue
        for feature_line in feature_lines:
            feature_line = feature_line.replace("\n","")
            op_feature = feature_line.split(",")
            fea_kind = op_feature[0]
            op_feature = [float(x) for x in op_feature[1:]]
            i = fea_kinds.find(fea_kind)
            if i < 0 :
                print(fea_kind," not use")
            else:
                op_features[i].extend(op_feature)

with open(center_path+"centermean.txt","w") as ffile:
    for i in range(len(op_features)):
        op_f = np.array(op_features[i])
        op_f = op_f.reshape(-1,fea_length[i])
        op_f[op_f>10] = 0
        mu = np.mean(op_f, axis=0)
        sigma = np.std(op_f, axis=0)
        op_f = (op_f - mu) / (sigma+1e-4)
        # op_fs.append(op_f)
        centermean = []
        centermean.extend(mu.tolist())
        centermean.extend(sigma.tolist())
        f_centers.append(centermean)
        centermean= [str(round(x,4)) for x in centermean]
        str_convert = ','.join(centermean)
        ffile.write(str_convert+"\n")
        print(fea_kinds[i]+"LP centermean done")

# clustering

# for i in range(len(op_fs)):
#     codebook, distortion = kmeans(op_fs[i], cluster_num)
#     print(codebook,distortion)
#     pickle.dump([codebook,distortion],open(center_path+str(train_bound)+'_'+fea_kinds[i]+'LP'+'.txt', 'wb') )
#     print(fea_kinds[i]+"LP clystering center done")

print("*****center files create successfully*****")

# create histograms

distortions = []

# train_histogram = histogram_path+"0_"+str(train_bound) +"_train_"
# gallery_histogram = histogram_path+str(train_bound+1) +"_124_gallery_"
# probe_histogram = histogram_path+str(train_bound+1) +"_124_probe_"

for view in views:

    for dtset_index in  [train_index,gallery_index,probe_index]:

        view_histograms = []

        with open(dtset_index+view+".txt","r") as ffile:
            #index_lines = ffile.readlines()
            index_lines = ffile.readlines()
            
        for index_line in index_lines:
            op_features = [[],[],[],[],[]]
            index_line = index_line.replace("\n","")
            if os.path.exists(index_line):
                with open(index_line,"r") as ffile:
                    feature_lines = ffile.readlines()[1:]
            else:
                print(index_line," file not found")
                continue

            for feature_line in feature_lines:
                feature_line = feature_line.replace("\n","")
                op_feature = feature_line.split(",")
                fea_kind = op_feature[0]
                op_feature = [float(x) for x in op_feature[1:]]
                i = fea_kinds.find(fea_kind)
                if i < 0 :
                    print(fea_kind," not use")
                else:
                    op_features[i].extend(op_feature)

            histograms = []

            for i in range(len(op_features)):
                mu = np.array(f_centers[i][:fea_length[i]])
                sigma = np.array(f_centers[i][fea_length[i]:])
                op_f = np.array(op_features[i])
                op_f = op_f.reshape(-1,fea_length[i])
                op_f = (op_f - mu) / (sigma+1e-4)
                with open(center_path+str(train_bound)+'_'+fea_kinds[i]+'LP'+'.txt','rb') as ffile:
                    codebook, distortion = pickle.load(ffile)
                    codebook = np.squeeze(np.array(codebook))
                histogram = np.zeros((1,cluster_num))
                label,dist = vq(op_f,codebook)

                for j in label:
                    histogram[0][j]+=1
                
                mu = np.mean(histogram, axis=1)
                sigma = np.std(histogram, axis=1)
                histogram = (histogram - mu) / (sigma+1e-4)
                histogram = np.ravel(histogram)
                histograms.append(histogram)

            view_histograms.append(histograms)
            #print(index_line," create histogram successfully")

        view_histograms = np.array(view_histograms)
        dtset_histogram = histogram_path + dtset_index.replace(index_path,"")
        print(dtset_histogram+view+" histograms shape: ",view_histograms.shape)
        pickle.dump(view_histograms,open(dtset_histogram+view+'_histogram.txt', 'wb') )

print("*****histogram files create successfully*****")




