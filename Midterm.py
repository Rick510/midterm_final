#先載入會使用到的資料庫
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

#將training 資料導入並定義df
df = pd.read_csv("C:\\Users\\rick\\AI code\\midterm\\train.csv")
#觀察資料的架構
df.head()
df.describe().T
#查看資料有無缺失值，並得知每個欄位的資料type(可發現此資料已經過初步結構化)
df.info()
#利用此行程式確認資料是否為imbalanced data(結果顯示還算平衡)
#若結果為imbalanced data 可能要用oversampling 或 undersampling的方式去平衡資料
df['target'].value_counts()
#在這張圖中能發現gravity越接近平均值被判為一個機率更高
sns.pairplot(df[['target','gravity']],dropna=True)

#觀察各欄資料與target欄位的關聯，發現calc欄位與target欄位可能最有關係
df.groupby('target').mean()
df.corr()
'''
藉由剛剛的程式已能知曉此資料集無須補空值。
而共只有五欄能用來分類，故全部欄位都會拿來預測結果。

'''
#創立模型的訓練集及測試集
#dx 就是將要分類的target去除後剩餘的其他欄位，dy則是要分類的目標
dx = df.drop(['target'], axis = 1)
dy = df['target']
print(dx.head())
print(dy.head())
#導入要使用的切分library，30%當測試集，且使用statify讓兩類資料在測試集平衡
from sklearn.model_selection import train_test_split
dx_train, dx_test, dy_train, dy_test = train_test_split(dx, dy, test_size = 0.3, random_state = 42,stratify = dy)


#載入第一個model(logisticregression)
from sklearn.linear_model import LogisticRegression
#定義模型
lr = LogisticRegression()
#將訓練集放入模型裡測試
lr.fit(dx_train, dy_train)
#讓訓練後的模型去分類測試集
predictions_lr=lr.predict(dx_test)
#載入能評價模型預測結果的matrix
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
#因資料集沒有不平等的問題，故等等主要還是使用準確度與其他模型對比
accuracy_score(dy_test,predictions_lr) 
recall_score(dy_test, predictions_lr) #怕Type II Error的，可使用此評價
precision_score(dy_test, predictions_lr) #怕Type I Error的，可使用此評價
#畫出confusion matrix
pd.DataFrame(confusion_matrix(dy_test, predictions_lr), columns=['Predict 0', 'Predict 1'], index=['True 0','True 1 '])


#載入第二個model(決策樹)
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(dx_train, dy_train)
predictions_dt=dt.predict(dx_test)
#發現decision tree 的預測結果比 logisticregression還要好
accuracy_score(dy_test,predictions_dt) 
recall_score(dy_test, predictions_dt) #怕Type II Error的，可使用此評價
precision_score(dy_test, predictions_dt) #怕Type I Error的，可使用此評價


#載入第三個模型(random forest)
#由於random forest是decision tree的進階版，一般來說會預期分類結果比decision tree好
from sklearn.ensemble import RandomForestClassifier
#因為怕子樹太多可能過度擬和或拖慢運算速度，故設定上限為100顆
RFC=RandomForestClassifier(n_estimators=100)
RFC.fit(dx_train, dy_train)
predictions_rfc=RFC.predict(dx_test)
#果然與預期一樣，隨機森林比之前兩個模型都好
accuracy_score(dy_test,predictions_rfc) 
recall_score(dy_test, predictions_rfc) #怕Type II Error的，可使用此評價
precision_score(dy_test, predictions_rfc) #怕Type I Error的，可使用此評價


#載入第四個模型(KNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(dx_train, dy_train)
predictions_knn = knn.predict(dx_test)
accuracy_score(dy_test,predictions_knn) 
recall_score(dy_test, predictions_knn) #怕Type II Error的，可使用此評價
precision_score(dy_test, predictions_knn) #怕Type I Error的，可使用此評價



#載入第五個模型(SVM)   
#一般SVM是在maching learning中預測能力最好的模型 我們使用此資料集驗證看看
from sklearn.svm import SVC
svm = SVC()
svm.fit(dx_train, dy_train)
predictions_svm = svm.predict(dx_test)
accuracy_score(dy_test,predictions_svm) #跟預期的效果有差異
recall_score(dy_test, predictions_svm) #怕Type II Error的，可使用此評價
precision_score(dy_test, predictions_svm) #怕Type I Error的，可使用此評價




#將五個不同模型的準確度印出
#看結果可發現RANDOM FOREST 表現最好因此拿它當比賽版本
print('accuracy of logisticregression:', accuracy_score(dy_test,predictions_lr),'\n','accuracy of decision tree:', accuracy_score(dy_test,predictions_dt),'\n', 'accuracy of random forest:', accuracy_score(dy_test,predictions_rfc), '\n', 'accuracy of KNN:',accuracy_score(dy_test,predictions_knn), '\n',  'accuracy of svm:', accuracy_score(dy_test,predictions_svm)   )

'''
可以看到跟預期的不一樣，SVM的分類效果不是最好的。
因此最後將RANDOM forest拿出來當最這次比賽的模型，
不過在比賽前，先尋找最佳參數來最大程度的提升準確度
'''

#但在送出前，進行參數優化來增加準確度
#參數優化一般有兩個套件選擇，optuna或是grid search。
#自己習慣在maching learning使用grid search，NN才使用optuna

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#設定參數的範圍、選項，讓程式去尋找最佳參數

para_grid = {'max_features' : ['auto', 'sqrt', 'log2'],'max_depth':[5,10,15,20],'n_estimators' : [10,30, 50, 100, 150, 200],'min_samples_leaf' : [10,30, 50,100]}
model = GridSearchCV(RandomForestClassifier(), para_grid)
#找最佳參數的話，找越多的參數、越大的範圍所需的時間就越多，但準確度也會上升
model.fit(dx_train, dy_train)
#利用此行程式印出最佳參數
print('best params:', model.best_params_)
#這行會印出最佳參數帶入後預測測試集的準確度
print('test score:', model.score(dx_test, dy_test).round(3))

#再將模型拿過來並使用最佳參數跑模型

RFC_final=RandomForestClassifier(max_depth=15, max_features='sqrt',min_samples_leaf=30,n_estimators=100)
RFC_final.fit(dx_train, dy_train)
predictions_rfc_final=RFC_final.predict(dx_test)
#果然與預期一樣，隨機森林比之前兩個模型都好
accuracy_score(dy_test,predictions_rfc_final) 
accuracy_score(dy_test,predictions_rfc) 
#將調參數前與調參數後做比較，發現調完後的確準確度有提升一些
print('the accuarcy before tunning parameter',accuracy_score(dy_test,predictions_rfc), '\n','the accuarcy after tunning parameter', accuracy_score(dy_test,predictions_rfc_final) )

#將最後的model打包輸出
import joblib
joblib.dump(RFC_final,'Classification.pk1',compress=3)
