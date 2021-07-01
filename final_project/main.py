from DataCleaner import *
from classifcation import*
from HandelMissingValues import*
credits_file="samples_tmdb_5000_credits_test.csv"
movies_reg="samples_tmdb_5000_movies_testing_regression.xlsx"
movies_class="samples_tmdb_5000_movies_testing_classification.xlsx"
#for classification

train, test=runDataCleaner(movies_class,credits_file , 'c')
X_train,X_test,y_train,y_test,features=handle_test_and_train_missings(train, test,'c')
#X_train,X_test,y_train,y_test= train_test_split(X_train, y_train, test_size=0.2 ,random_state=42 )

print("classification Without PCA")
classification_withoutPCA(X_train,X_test,y_train,y_test,0)

print("classificstion With PCA")
classification_with_PCA(X_train,X_test,y_train,y_test,1)


#for regression:

train, test=runDataCleaner(movies_reg,credits_file , 'r')

X_train,X_test,y_train,y_test,features=handle_test_and_train_missings(train, test,'r')
regression(X_train,X_test,y_train,y_test,features)




