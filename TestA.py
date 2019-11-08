#Test distance computation in a matrix
#Miniature Dope

#import modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

sns.set(style = "whitegrid")
    
class Distance_Entropy:
    def __init__(self, data_source):
        self.data_source = data_source
        self.data = pd.read_csv(data_source)
        

    def Columns(self):
        Col_Names = self.data.columns
        return Col_Names

    #Compute entropy of the whole dataframe
    def Entropy_A(self):
        Entropy1 = scipy.stats.entropy(self.data)
        Entropy1 = np.reshape(Entropy1, (1, len(self.Columns())))
        EntropDF = pd.DataFrame(data = Entropy1, columns = self.Columns())
        EntropDF = EntropDF.T
        EntropDF.columns = ["EntA"]
        return EntropDF

    #Breaking the data in columns of data
    def ExtractData(self):
        Items2 = []
        for x,y in self.data.iteritems():
            List = list(y)          #Convert the pandas series to list for easy manipulation
            Items2.append(List)
        return Items2

             
    #Calculate Distance
    def Distance_A(self,x,y): 
        return abs(x-y)
    
    def Distance_B(self,x,y):  #This type of distance doesnt use absolute
        return (x-y)
            

    #Build the distance matrix
    # This function only calculates for one (n) column
    def Matrix_A(self,n):
        Current_Data = self.ExtractData()
        Current_Data = Current_Data[n]
        features = len(Current_Data)
        Matrix = np.zeros((features, features))
        for index_i, col_i in enumerate(Current_Data):
            for index_j, col_j in enumerate(Current_Data):
                Matrix[index_i, index_j] = self.Distance_A(col_i, col_j)
        Col = Current_Data
        P_Matrix = pd.DataFrame(data = Matrix, columns = Col, index = Col)
        return P_Matrix

    def Matrix_B(self,n):
        Current_Data = self.ExtractData()
        Current_Data = Current_Data[n]
        features = len(Current_Data)
        Matrix = np.zeros((features, features))
        for index_i, col_i in enumerate(Current_Data):
            for index_j, col_j in enumerate(Current_Data):
                Matrix[index_i, index_j] = self.Distance_B(col_i, col_j)
        Col = Current_Data
        P_Matrix = pd.DataFrame(data = Matrix, columns = Col, index = Col)
        return P_Matrix

    #This function calculates for all columns and save in a database
    def All_Data_B(self):
        Len = len(self.Columns())
        Final_Mat = []
        for i in range(Len):
            Final_Mat.append(self.Matrix_A(i))
        Final_Data = dict(zip(self.Columns(), Final_Mat))
        return Final_Data

    
    def All_Data_C(self):
        Len = len(self.Columns())
        Final_Mat = []
        for i in range(Len):
            Final_Mat.append(self.Matrix_B(i))
        Final_Data = dict(zip(self.Columns(), Final_Mat))
        return Final_Data

    # Compute Entropy accross Database
    def Entropy_B(self):
        DATA = self.All_Data_B()
        Ents = []
        Keys = []
        for key in DATA.keys():
            Ents.append(scipy.stats.entropy(DATA[str(key)]))
            Keys.append(key)
        Dict = {}
        for i in range(len(Ents)):
            Dict[(Keys[i])] = sum(Ents[i])
        DataFrame = pd.DataFrame(list(Dict.items()), index = self.Columns())
        DataFrame = DataFrame.T
        DataFrame = DataFrame.drop([0])
        DataFrame = DataFrame.T
        DataFrame.columns = ["EntB"]
        return DataFrame

    def Entropy_C(self):
        DATA = self.All_Data_C()
        Ents = []
        Keys = []
        for key in DATA.keys():
            Ents.append(scipy.stats.entropy(DATA[str(key)]))
            Keys.append(key)
        Dict = {}
        for i in range(len(Ents)):
            Dict[(Keys[i])] = sum(Ents[i])
        DataFrame = pd.DataFrame(list(Dict.items()), index = self.Columns())
        DataFrame = DataFrame.T
        DataFrame = DataFrame.drop([0])
        DataFrame = DataFrame.T
        DataFrame.columns = ["EntC"]
        return DataFrame

    def Final_Ent_Data(self):
        All_Data = self.Entropy_A()
        All_Data["EntB"] = self.Entropy_B()["EntB"]
        All_Data["EntC"] = self.Entropy_C()["EntC"]
        return All_Data

    def Normal_Final_Ent_Data(self):
        DataF = pd.DataFrame()
        
        Min_A = min(self.Final_Ent_Data()["EntA"])
        Max_A = max(self.Final_Ent_Data()["EntA"])
        DataF["Normal_EntA"] = [((i -Min_A)/(Max_A - Min_A)) for i in self.Final_Ent_Data()["EntA"]]

        Min_B = min(self.Final_Ent_Data()["EntB"])
        Max_B = max(self.Final_Ent_Data()["EntB"])
        DataF["Normal_EntB"] = [((i -Min_B)/(Max_B - Min_B)) for i in self.Final_Ent_Data()["EntB"]]

        Min_C = min(self.Final_Ent_Data()["EntC"])
        Max_C = max(self.Final_Ent_Data()["EntC"])
        DataF["Normal_EntC"] = [((i -Min_C)/(Max_C - Min_C)) for i in self.Final_Ent_Data()["EntC"]]
        return DataF
    
    #Plot for further analysis    
    def PlotBar(self):
        Data = self.Entropy_DataFrame().T
        X = Data.columns
        Y = Data.iloc[0]
        plt.bar(X,Y)
        plt.show()

    def Histogram(self):
        Data = self.Entropy_DataFrame().T
        X = Data.columns
        Y = Data.iloc[0]
        plt.hist(Y, bins = len(self.Columns()))
        plt.show()

    def NormHist(self):
        Data = self.Entropy_DataFrame().T
        X = Data.iloc[0]
        x_std = np.std(X)
        x_mean = np.mean(X)
        Y = scipy.stats.norm.pdf(list(X), x_mean, x_std)
        plt.plot(X,Y)
        plt.hist()
        plt.show()
        
