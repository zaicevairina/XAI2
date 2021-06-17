
from catboost import CatBoostClassifier,Pool
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_model(name,path,params):
        df = pd.read_csv(path)
        y = df.target
        X = df.drop('target', axis=1)
        model = CatBoostClassifier()
        params['train_dir']=name
        model.set_params(**params)
        feature_names = list(X.columns)
        pool = Pool(X, y, cat_features=params['cat_features'], feature_names=feature_names)

        model.fit(pool,verbose=False)

        with open(f'{name}/{name}.pickle', 'wb') as f:
            pickle.dump(model, f)

        with open(f'{name}/{name}_data.pickle', 'wb') as f:
            pickle.dump(df, f)

        s = 'Число деревьев: ' + str(model.tree_count_)
        print(s)

        return (s)

def predict(name,path,obj=None):

     ######
     obj = [63, 1, 3,145, 230, 0, 1, 168, 1, 0, 2, 0, 3]
     #########

     with open(f'{name}/{name}.pickle', 'rb') as f:
         model = pickle.load(f)

     with open(f'{name}/{name}_data.pickle', 'rb') as f:
         df = pickle.load(f)


     #Probability
     Probability = model.predict(obj,prediction_type='Probability')
     #Class
     Class = model.predict(obj,prediction_type='Class')
     #RawFormulaVal
     RawFormulaVal = model.predict(obj,prediction_type='RawFormulaVal')

     print(Probability)
     print(Class)
     print(RawFormulaVal)
     return (Probability,Class,RawFormulaVal)





def predict_some_trees(name,ntree_start,ntree_end,path,obj=None):
     with open(f'{name}/{name}.pickle', 'rb') as f:
         model = pickle.load(f)

     ######
     obj = [63, 1, 3,145, 230, 0, 1, 168, 1, 0, 2, 0, 3]

     #########

     #Probability
     Probability = model.predict(obj,prediction_type='Probability',ntree_start=ntree_start,ntree_end=ntree_end)
     #Class
     Class = model.predict(obj,prediction_type='Class',ntree_start=ntree_start,ntree_end=ntree_end)
     #RawFormulaVal
     RawFormulaVal = model.predict(obj,prediction_type='RawFormulaVal',ntree_start=ntree_start,ntree_end=ntree_end)

     print(Probability)
     print(Class)
     print(RawFormulaVal)

     return (Probability,Class,RawFormulaVal)




def show_tree(name,id):
    with open(f'{name}/{name}.pickle', 'rb') as f:
        model = pickle.load(f)

    with open(f'{name}/{name}_data.pickle', 'rb') as f:
        df = pickle.load(f)


    feature_names = list(df.columns)[:-1].extend('tatget')
    cat_features = model._get_params()['cat_features']
    X = df
    pool_tr = Pool(X, cat_features=cat_features, feature_names=feature_names)
    model.plot_tree(tree_idx=id,pool=pool_tr,path=f'media/{name}_tree_{id}')
    path = f'/media/{name}_tree_{id}.png'
    return path




def XAI(name,path,obj=None):

     ######
    obj = [63, 1, 3,145, 230, 0, 1, 168, 1, 0, 2, 0, 3]


    #########

    with open(f'{name}/{name}.pickle', 'rb') as f:
        model = pickle.load(f)

    with open(f'{name}/{name}_data.pickle', 'rb') as f:
        df = pickle.load(f)


    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values([obj])
    with open(f'{name}/{name}_data.pickle', 'rb') as f:
        df = pickle.load(f)

    feature_names = list(df.columns)
    feature_names.remove('target')

    shap.force_plot(explainer.expected_value, shap_values[0,:],
                       features=np.array(obj),feature_names=feature_names)

    shap.save_html(f'media/{name}'+'_shap_1.html', shap.force_plot(explainer.expected_value, shap_values[0,:],
                   features=np.array(obj),feature_names=feature_names))
    path = f'/media/{name}'+'_shap_1.html'
    return path

def info(name):

     with open(f'{name}/{name}.pickle', 'rb') as f:
         model = pickle.load(f)

     with open(f'{name}/{name}_data.pickle', 'rb') as f:
         df = pickle.load(f)

     count_tree = 'Число деревьев в ансамбле '+ str(model.tree_count_)
     train_size = 'Размер обучающей выборки ' + str(df.shape[0])

     y = df.target
     X = df.drop('target', axis=1)

     feature_names = list(df.columns)[:-1].extend('tatget')
     cat_features = model._get_params()['cat_features']

     y = df.target
     X = df.drop('target', axis=1)

     pool_tr = Pool(X,y ,cat_features=cat_features, feature_names=feature_names)
     feature_importance_PredictionValuesChange = model.get_feature_importance(pool_tr,type='PredictionValuesChange',prettified=True)

     feature_importance_LossFunctionChange = model.get_feature_importance(pool_tr,type='LossFunctionChange',prettified=True)


     x = model.get_params()
     params = []
     for param,value in x.items():
         if param!='train_dir':
            if isinstance(value,list):
              s = param + ': ' + ', '.join(value)
            else:
              s = param + ': ' + str(value)
            params.append(s)



     feature_names = list(df.columns)
     feature_names.remove('target')
     feature_names = 'Признаки использующиеся в обучении ' +','.join(feature_names)


     return (feature_names,count_tree,train_size,feature_importance_PredictionValuesChange,feature_importance_LossFunctionChange,params)
    #     описнаие модели


    #     график

def interaction(name):
     with open(f'{name}/{name}.pickle', 'rb') as f:
         model = pickle.load(f)

     with open(f'{name}/{name}_data.pickle', 'rb') as f:
         df = pickle.load(f)
     y = df.target
     X = df.drop('target', axis=1)

     feature_names = list(df.columns)[:-1].extend('tatget')
     cat_features = model._get_params()['cat_features']

     pool_tr = Pool(X,y ,cat_features=cat_features, feature_names=feature_names)

     fi = model.get_feature_importance(pool_tr, type="Interaction")

     fi_new = []
     for k,item in enumerate(fi):
         first = X.dtypes.index[fi[k][0]]
         second = X.dtypes.index[fi[k][1]]
         if first != second:
            fi_new.append([first + "_" + second, fi[k][2]])

     feature_score = pd.DataFrame(fi_new,columns=['Feature-Pair','Score'])
     feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
     plt.rcParams["figure.figsize"] = (16,7)
     ax = feature_score.plot('Feature-Pair', 'Score', kind='bar', color='c')
     ax.set_title("Pairwise Feature Importance", fontsize = 14)
     ax.set_xlabel("features Pair")
     path = (f'/media/{name}_pair.png')
     plt.savefig(f'media/{name}_pair.png', format='png', bbox_inches = 'tight')
     plt.show()
     plt.close()
     return path

def shap_summary(name):

     with open(f'{name}/{name}.pickle', 'rb') as f:
         model = pickle.load(f)

     with open(f'{name}/{name}_data.pickle', 'rb') as f:
         df = pickle.load(f)

     del df['target']
     explainer = shap.TreeExplainer(model)

     observations = df.to_numpy()
     shap_values = explainer.shap_values(observations)

     feature_names = list(df.columns)

     shap.summary_plot(shap_values, features=observations, feature_names=feature_names, show=False)
     path = f'/media/{name}_shap_summary.png'
     plt.savefig(f'media/{name}_shap_summary.png', format='png', bbox_inches = 'tight')
     plt.close()
     return path
