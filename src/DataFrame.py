import pandas as pd
import os

class DF():
    def __init__(self, path='dataset2.csv', remove = False):
        self.df = pd.read_csv(path)
        self.data = {}
        for att in self.df:
            if remove:
                if att not in remove:
                    self.data[att] = list(self.df.get(att)) 
            else:
                self.data[att] = list(self.df.get(att)) 
        self.index = {i:col for i, col in zip(range(len(self.data.keys())), list(self.data.keys()))}  
        self.reverse_index = {col:i for i, col in zip(range(len(self.data.keys())), list(self.data.keys()))}  

    def encode_str(self):
        data_to_encode = {}
        for k, v in self.data.items():
            if (type(v[0]) == str):
                data_to_encode.update({k:v})
        unique = [[self.unique(v) for v in data_to_encode.values()]][0]
        for i, k, v in zip(range(len(unique)), list(data_to_encode.keys()), list(data_to_encode.values())):
            data_to_encode[k] = [unique[i].index(l) for l in v if l != 'nan']  
        self.encode_dict = {k: {key: val for val, key in zip(v, [i for i in range(len(v))])} for k, v in zip(list(data_to_encode.keys()), unique)}
        self.data.update(data_to_encode)

    def unique(self, l):
        l = self.sorted(l)
        unique = []
        for x in l:
            if x not in unique:
                unique.append(x) 
        return unique

    def mean_list(self, l):
        return sum(l)/len(l)

    def where(self, col, cond): 
        if type(col) == str:
            col = self.data.get(col)
        return [index for index, x in enumerate(col) if eval(cond)]

    def reduce_w(self, col, bins):
        col = self.sorted(col) 
        w = int((max(col) - min(col)) / bins)
        min1 = min(col)
        arr = []
        for i in range(0, bins + 1):
            arr = arr + [min1 + w * i]
        arri=[]
        for i in range(0, bins):
            temp = []
            for j in col:
                if j >= arr[i] and j <= arr[i+1]:
                    temp += [j]
            arri += [temp] 
        l =  [['['+str(i[0])+', '+str(i[-1])+']']*len(i) for i in arri]
        ret = []
        for i in l:
            ret = ret + i
        return ret

    def reduce_f(self, col, bins):
        col = self.sorted(col) 
        a = len(col)
        n = int(a / bins)
        new = []
        for i in range(0, bins):
            arr = [] 
            for j in range(i * n, (i + 1) * n):
                if j >= a:
                    break
                arr = arr + [col[j]]
            new.append(arr) 
        l =  [['['+str(i[0])+', '+str(i[-1])+']']*len(i) for i in new]
        ret = []
        for i in l:
            ret = ret + i
        return ret

    def count(self, col):
        if type(col) == str:
            col = self.data.get(col)
        return {k: col.count(k) for k in self.unique(col)}

    def sorted(self, col):
        if type(col) == str:
            col = self.data.get(col)
        return sorted([i for i in col if str(i) != 'nan'])

    def Q(self, q, col):
        if (type(self.data.get(col)[0]) == str):
            return None
        col = self.sorted(col)
        return col[int(len(col)*(q/4))]

    def min(self, col):
        if (type(self.data.get(col)[0]) == str):
            return None
        return self.sorted(col)[0]

    def max(self, col):
        if (type(self.data.get(col)[0]) == str):
            return None
        return self.sorted(col)[-1]

    def mean(self, col):
        try:
            if (type(self.data.get(col)[0]) == str):
                return None
        except:
            pass
        col = self.sorted(col)
        return self.mean_list(col)

    def variance(self, col):
        col = self.sorted(col)
        mean = self.mean_list(col)
        return sum((x - mean) ** 2 for x in col) / len(col)
   
    def std(self, col):
        return self.variance(col)**0.5

    def summary(self, col):
        return {'mean':self.mean(col), 'std':self.std(col), 'min':self.min(col), 'Q1':self.Q(1, col), 'M':self.Q(2, col), 'Q3':self.Q(3, col), 'max':self.max(col)}

    def outliers(self, col):
        IQR = (self.Q(3, col) - self.Q(1, col)) * 1.5
        minlist = self.where(self.data.get(col), 'x<%s' % (self.Q(1, col)-IQR))
        maxlist = self.where(self.data.get(col), 'x>%s' % (self.Q(3, col)+IQR))
        return {'Min outliers': {k:self.data.get(col)[k] for k in minlist}, 'Max outliers': {k:self.data.get(col)[k] for k in maxlist}}

    def outliers_median_imputation(self, col_name):
        outliers = self.outliers(col_name)
        col = self.sorted(col_name)
        med = col[int(len(col)*(0.5))]
        outliers = list(outliers['Min outliers'].keys()) + list(outliers['Max outliers'].keys())
        for out in outliers:
            self.data.get(col_name)[out] = med

    def normalize_minmax(self, col):
        self.encode_str()
        min = self.min(col)
        max = self.max(col)
        for i in range(len(self.data.get(col))):
            self.data.get(col)[i] = (self.data.get(col)[i]-min)/(max-min)

    def normalize_meanstd(self, col):
        self.encode_str() 
        std = self.std(col) 
        mean = self.mean(col) 
        for i in range(len(self.data.get(col))):
            self.data.get(col)[i] = (self.data.get(col)[i]-mean)/std

    def missing_values_imputation(self):
        for col_name in list(self.data.keys()):
            col = self.data.get(col_name)
            med = col[int(len(col)*(0.5))]
            for i in range(len(self.data.get(col_name))):
                if (str(self.data.get(col_name)[i]) == 'nan'):
                    self.data.get(col_name)[i] = med

