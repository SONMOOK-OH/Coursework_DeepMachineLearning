import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import preprocessing


class FN_KNN(object):

    def __init__(self, lam=0.1, w1=0.5, w2=0.5, N=1000, rho=0.1, pi=0.5, P_conv=0.01, k=1):

        self.lam = lam
        self.w1 = w1
        self.w2 = w2
        self.N = N
        self.rho = rho
        self.pi = pi
        self.P_conv = P_conv
        self.k = k

        self.X_train = None
        self.y_train = None

    def get_label(self, query, X, y, k):

        L1 = self.compute_L1(query=query, data=X)
        print('L1 is done')
        L2 = self.compute_L2(query=query, data=X, lam=self.lam)
        print('L2 is done')
        R = self.MC_CE(L1=L1, L2=L2, w1=self.w1, w2=self.w2, N=self.N, rho=self.rho, pi=self.pi, P_conv=self.P_conv)

        y_knn = y[R[:k]]

        label = np.argmax(np.bincount(y_knn))

        return label

    def fit(self, X_train, y_train):

        self.X_train = np.array(X_train, dtype=np.float32)
        self.y_train = np.array(y_train, dtype=np.int32)

    def score(self, X_test, y_test):

        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        y_pred = np.zeros(shape=y_test.shape, dtype=np.int32)

        for i in range(X_test.shape[0]):

            L1 = self.compute_L1(query=X_test[i, :], data=self.X_train)
            print('L1 is done')
            print('L1:', L1)

            L2 = self.compute_L2(query=X_test[i, :], data=self.X_train, lam=self.lam)
            print('L2 is done')
            print('L2:', L2)

            R = self.MC_CE(L1=L1, L2=L2, w1=self.w1, w2=self.w2, N=self.N, rho=self.rho, pi=self.pi, P_conv=self.P_conv)
            print('R is done')
            print('R:', R)

            y_knn = self.y_train[R[:self.k]]

            y_pred[i] = np.argmax(np.bincount(y_knn))

        accuracy = np.mean(y_pred == y_test)

        return accuracy

    def compute_L1(self, query, data):

        L1_distance = []

        for j in range(data.shape[0]):
            d = np.linalg.norm(query - data[j, :])  # euclidean distance ( = 2-norm)
            L1_distance.append(d)

        L1 = np.argsort(L1_distance)[:self.k]  # ordered list L1

        return L1
        
    def compute_L2(self, query, data, lam):

        L2_distance = np.zeros(shape=(data.shape[0],))

        U = np.concatenate((query.reshape(1, -1), data), axis=0)

        U_n = U.shape[0]

        for a in range(U.shape[1]):

            list_n = []

            for i in range(U_n):

                n = 0

                for j in range(U_n):

                    if np.abs(U[i, a] - U[j, a]) <= lam:

                        n += 1

                list_n.append(n/U_n)

                if i != 0:

                    L2_distance[i-1] += np.abs(list_n[0] - list_n[-1])**2

        L2 = np.argsort(L2_distance)[:self.k] # ordered list L2

        return L2

    def footrule_distance(self, R, L):

        check = []

        for i in np.concatenate((R, L), axis=0):
            if i not in check:
                check.append(i)


        d = 0

        for t in check:

            if t not in R:
                R_rank = len(R)
            else:
                R_rank = np.where(R == t)[0][0]

            if t not in L:
                L_rank = len(L)
            else:
                L_rank = np.where(L == t)[0][0]

            d += np.abs(R_rank - L_rank)

        return d

    def cost_function(self, R, L1, L2, w1, w2):

        cost = w1 * self.footrule_distance(R=R, L=L1) + w2 * self.footrule_distance(R=R, L=L2)

        return cost

        
    def MC_CE(self, L1, L2, w1, w2, N,  rho, pi, P_conv):

        check = []

        for i in np.concatenate((L1, L2), axis=0):
            if i not in check:
                check.append(i)

        check = np.array(check, dtype=np.int32)

       

        n_row = len(check)
        n_col = len(L1)

        num_best = int(np.around(N * rho))

        P = np.ones((n_row, n_col)) / n_row

        X = np.zeros((N, n_row, n_col))

        R = np.zeros((N, n_col), dtype=np.int32)

        cost_value = np.zeros((N,))

        P_mean = np.inf
        
        while P_mean >= P_conv:

            for i in range(N):

                #print('MCCE sampling :', i)

                while True:

                    for k in range(n_col):

                        X[i, :, k] = np.random.multinomial(n=1, pvals=P[:, k])

                    cond = np.sum(X[i, :, :], axis=1)

                    if (cond <= 1).all():
                        R_ind = np.argmax(X[i, :, :], axis=0)
                        R[i, :] = check[R_ind]
                        cost_value[i] = self.cost_function(R=R[i, :], L1=L1, L2=L2, w1=w1, w2=w2)
                        break
                    
            y = np.sort(cost_value)[num_best]

            best_idx = np.where(cost_value <= y)[0]
            
            P_new = np.sum(X[best_idx, :, :], axis=0) / len(best_idx)

            P_mean = np.mean(np.abs(P - P_new))

            P = (1-pi) * P + pi * P_new

        R_ind = np.argmax(P, axis=0)
        R_star= check[R_ind]

        return R_star
    
    
        
def Data_Generation(k):
    
    if k==1:
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
        names=["erythema","scaling","definite borders","itching","koebner phenomenon","polygonal papules"
             ,"follicular papules","oral mucosal involvement","knee and elbow involvement","scalp involvement","family history","melanin incontinence"
             ,"eosinophils in the infiltrate","PNL infiltrate","fibrosis of the papillary dermis","exocytosis","acanthosis","hyperkeratosis","parakeratosis"
             ,"clubbing of the rete ridges","elongation of the rete ridges","thinning of the suprapapillary epidermis","spongiform pustule"
             ,"munro microabcess","focal hypergranulosis","disappearance of the granular layer","vacuolisation and damage of basal layer","spongiosis"
             ,"saw-tooth appearance of retes","follicular horn plug","perifollicular parakeratosis","inflammatory monoluclear inflitrate","Age","band-like infiltrate", "Class"]
        data=pd.read_csv(url,header=-1,names=names)
        for i in range (0, len(names)):
            data1=data[names[33]].replace('?','30')
            data.drop([names[33]],inplace=True,axis=1)
            data=pd.concat([data,data1],axis=1)
        X,Y=data.ix[:,0:35],data.ix[:,33]
        X.drop([names[34]],inplace=True,axis=1)
        Attributes=["erythema","scaling","definite borders","itching","koebner phenomenon","polygonal papules"
                     ,"follicular papules","oral mucosal involvement","knee and elbow involvement","scalp involvement","family history","melanin incontinence"
                     ,"eosinophils in the infiltrate","PNL infiltrate","fibrosis of the papillary dermis","exocytosis","acanthosis","hyperkeratosis","parakeratosis"
                     ,"clubbing of the rete ridges","elongation of the rete ridges","thinning of the suprapapillary epidermis","spongiform pustule"
                     ,"munro microabcess","focal hypergranulosis","disappearance of the granular layer","vacuolisation and damage of basal layer","spongiosis"
                     ,"saw-tooth appearance of retes","follicular horn plug","perifollicular parakeratosis","inflammatory monoluclear inflitrate","Age","band-like infiltrate"]    

            
    elif k==2: 
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        names=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
        data=pd.read_csv(url,header=-1,names=names)
        X,Y=data.ix[:,0:9],data.ix[:,9]
        Attributes=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
        
    elif k==3:
        url='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'
        names=['age','sex','chest_pain_type','resting_blood_pressure','serum_cholestoral_in_mg','fasting_blooding_sugar>120mg','resting_electrocardiographic_results'
                   , 'maximum_heart_rate_achieved','exercise_induced_angina','oldpeak','the_slope_of_the_peak_exercise_ST_segment'
                   , 'number_of_major_vessels_colored_by_flourosopy', 'thal', 'Class']
        data=pd.read_csv(url,sep=' ',header=-1,names=names)
        X,Y=data.ix[:,0:13],data.ix[:,13]
        Attributes=['age','sex','chest_pain_type','resting_blood_pressure','serum_cholestoral_in_mg','fasting_blooding_sugar>120mg','resting_electrocardiographic_results'
                   , 'maximum_heart_rate_achieved','exercise_induced_angina','oldpeak','the_slope_of_the_peak_exercise_ST_segment'
                   , 'number_of_major_vessels_colored_by_flourosopy', 'thal']
        
    elif k==4:
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
        names=["Class","AGE","SEX","STEROID","ANTIVIRALS","FATIGUE"
                     ,"MALAISE","ANOREXIA","LIVER BIG","LIVER FIRM","SPLEEN PALPABLE","SPIDERS"
                     ,"ASCITES","VARICES","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME","HISTOLOGY"]
        data=pd.read_csv(url,header=-1,names=names)

        for i in range (0, len(names)):
            if i in [1 , 2 , 4 ,19]:
                data1=data[names[i]].replace('?','0')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i in [3 , 5 , 6 , 7]:
                data1=data[names[i]].replace('?','1')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i in [10 , 11 , 12 , 13]:
                data1=data[names[i]].replace('?','5')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i==8:
                data1=data[names[i]].replace('?','10')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i==9:
                data1=data[names[i]].replace('?','11')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i==14:
                data1=data[names[i]].replace('?','6')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i==15:
                data1=data[names[i]].replace('?','29')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i==16:
                data1=data[names[i]].replace('?','4')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i==17:
                data1=data[names[i]].replace('?','16')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
            if i==18:
                data1=data[names[i]].replace('?','67')
                data.drop([names[i]],inplace=True,axis=1)
                data=pd.concat([data,data1],axis=1)
        X,Y=data.ix[:,1:],data.ix[:,0]
        Attributes=["AGE","SEX","STEROID","ANTIVIRALS","FATIGUE"
                     ,"MALAISE","ANOREXIA","LIVER BIG","LIVER FIRM","SPLEEN PALPABLE","SPIDERS"
                     ,"ASCITES","VARICES","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME","HISTOLOGY"]
        
    else:               
        data=pd.read_csv(r'Datasets\SRBCT.tab', sep='\t',header=2)#data route update!!!!!!!*****************
        data1=data['class'].replace('EWS','1')
        data1=data1[:].replace('BL','2')
        data1=data1[:].replace('NB','3')
        data1=data1[:].replace('RMS','4')
        Y=data1
        data.drop(['class'],inplace=True,axis=1)
        X=data
        Attributes=[]
    

    return (X, Y, Attributes) 
        
        
if __name__ == '__main__':
    
   
    knn = FN_KNN(k=5, w1=0.5, w2=0.5, N=1000)
    
    Total_acc=[]
    for k in range (1,5):
        X,Y, Attributes = Data_Generation(k)

    #Using mean normalization for numerical variables    
        min_max_scaler=preprocessing.MinMaxScaler()
        if k==5:
            X=pd.DataFrame(min_max_scaler.fit_transform(X.values), index=X.index)
        else:
            X=pd.DataFrame(min_max_scaler.fit_transform(X.values),columns=Attributes, index=X.index)
        
        
        jjj = 0
        Acc=[]
        Acc1=[]
        kfold= KFold(n_splits=10, shuffle=True,random_state=123)
        for train_index, test_index in kfold.split(X):
            jjj += 1
            print(jjj)
            print('----------')
            print("TRAIN:", train_index, "TEST:", test_index)
            if k==2:
                train_index=train_index+1
                test_index=test_index+1
            X_train, X_test= X.ix[train_index], X.ix[test_index]
            Y_train, Y_test = Y.ix[train_index], Y.ix[test_index]
            
    
            knn.fit(X_train, Y_train)
            acc = knn.score(X_test, Y_test)
            Acc.append(acc)
        
        Acc_mean = np.mean(Acc)
        Total_acc.append(Acc_mean)
    Final_acc=np.mean(Total_acc)
    
    
    
   
