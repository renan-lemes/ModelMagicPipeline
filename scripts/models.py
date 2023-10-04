import dataset

train, test = dataset.LoadData()

data_set = dataset.DataStruct(train, test).data_cat__num()

categoric_features, numeric_features = (
    data_set.catecoric_features,
    data_set.numeric_features,
)
