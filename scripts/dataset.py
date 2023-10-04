## Carregar dados
def LoadData():
    from pandas import read_csv

    train = read_csv("D:\\Projeto_\\ModelMagicPipeline\\data\\train.csv")
    test = read_csv("D:\\Projeto_\\ModelMagicPipeline\\data\\test.csv")
    return train, test


## Class para poder arrumar os dados de treino e teste dos modelos
class DataStruct:
    def __init__(self, train, test):
        self.test = test
        self.train = train

    def data_cat__num(self):
        self.catecoric_features = [
            col for col in train.columns if self.train[col].dtype.name == "object"
        ]
        self.numeric_features = [
            col
            for col in self.train.columns
            if col not in catecoric_features and col != "Survived"
        ]
