#!/usr/bin/python


import numpy
import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest , f_classif
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot
from texttable import Texttable
import operator

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'total_payments', 'exercised_stock_options','bonus',
                 'restricted_stock', 'total_stock_value','director_fees','expenses',
                 'loan_advances', 'deferred_income','restricted_stock_deferred',
                 'long_term_incentive','shared_receipt_with_poi']




def plot_data(sub_data):
    
    '''creates a scatterplot with the given data'''
    
    features = ["salary", "bonus"]
    data1 = featureFormat(sub_data, features)
    data1 = numpy.reshape( numpy.array(data1) , (len(data1),2))
    for point in data1:
        salary = point[0]
        bonus = point[1]
        matplotlib.pyplot.scatter( salary, bonus )

    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.show()




def make_NaN_to_Zero(dictvalue):
    
    ''' Assign zero to NaN values for specific features'''
    
    make_zero =['salary','total_stock_value','exercised_stock_options','from_this_person_to_poi',   'from_poi_to_this_person','from_messages','to_messages']
    for key in make_zero:
        if dictvalue[key]=='NaN':
            dictvalue[key]=0
    return dictvalue



def get_precision_recall_score(labels_test, pred):
    
    '''Get the Confusion matrix,precision and Recall scores'''
    
    
    print "CONFUSION MATRIX :\n",confusion_matrix(labels_test, pred)
    
    
    cpp = [1 for j in zip(labels_test, pred) if j[0] == j[1] and j[1] == 1]
    print "\n",cpp


    return (precision_score(labels_test, pred),recall_score(labels_test, pred))





result_list =[['Algorithm', 'Accuracy','Precision score','Recall score']]


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
    
    print "**********************************"
    print "DATA EXPLORATION\n"
    count=0
    True_poi_list =[]
    False_poi_list =[]
    

    print "Number of datapoints      :",len(data_dict)
    NaN_dict={}
    for data in data_dict:
        if count==0:
            count =count +1
            key_list = data_dict[data].keys()
            
            print "Number of Features        :" ,len(key_list)
            #print "Features are :",key_list
        
        
        if data_dict[data]["poi"]:
            True_poi_list.append(data)
        else:
            False_poi_list.append(data)
    
        for key in key_list:
            if data_dict[data][key]=='NaN':
                
                if key in NaN_dict.keys():
                    NaN_dict[key] = NaN_dict[key] +1
                else:
                    NaN_dict[key]= 1

print "Number of POIS            :", len(True_poi_list)
#print "POIS:\n" ,True_poi_list
print "Number of Non POIS        :", len(False_poi_list)
#print "Non POIS:\n" ,False_poi_list
print "**********************************"

print "\nCount of NAN per feature\n"


t = Texttable()
j=0

sorted_NaN = sorted(NaN_dict.items(), key=operator.itemgetter(1),reverse=True)

for i in sorted_NaN:
    if j==0:
        t.add_row(('Feature','Count of NaN'))
        j=j+1
                  
    t.add_row(i)

print t.draw()






### Task 2: Remove outliers


data_forplot = data_dict
plot_data(data_forplot)

NaN_salary_bonus =[]
print "Top 3 bonus and salary datapoints are :"
for key,value in data_dict.iteritems():
    if ((value['salary']=='NaN') and (value['bonus']=='NaN')):

        '''Gathering list of NaN salary and bonus datapoints'''
        NaN_salary_bonus.append(key)
    
    elif ((value['bonus']>= 5000000) and  (value['salary']>= 1000000)):

        print key


outlier =[]

'''
    # printing out the values of the NaN_salary_bonus list.
    print NaN_salary_bonus
    for person in  NaN_salary_bonus:
        print "\n",person
        print " \n" , data_dict[person]
'''



'''Datapoints that needs to be removed is added to a list '''

for key in data_dict.keys():
    if key=='TOTAL':
        outlier.append(key)
    elif key =="LOCKHART EUGENE E":
        outlier.append(key)
    elif key =="THE TRAVEL AGENCY IN THE PARK" :
        outlier.append(key)


print "\nRemoved Datapoints are:"
for key in outlier:
    data_dict.pop(key,0)
    print key



data_forplot = data_dict
plot_data(data_forplot)



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

for key,value in data_dict.iteritems():
    
    value =make_NaN_to_Zero(value)

    temp = value['salary']+value['total_stock_value']+value['exercised_stock_options']
    data_dict[key]['Total_earnings'] = temp
    poi_num = float(value['from_this_person_to_poi']+ value['from_poi_to_this_person'])
    poi_deno = float(value['from_messages']+ value['to_messages'])
    
    if (poi_deno > 0.0) and (poi_num >0.0) :
        
        data_dict[key]['poi_ratio'] = poi_num /poi_deno
    else:

        data_dict[key]['poi_ratio']=0




'''Printing out the number of features (including two new features)'''

print "\nTotal number of Features  :",len(data_dict['LAY KENNETH L'].values())

print "\nTotal number of datapoints :", len(data_dict)
my_dataset = data_dict



'''New features are added to the features list'''

features_list = ['poi','salary', 'total_payments', 'exercised_stock_options','bonus',
                 'restricted_stock', 'total_stock_value','director_fees','expenses',
                 'loan_advances', 'deferred_income','restricted_stock_deferred',
                 'long_term_incentive','shared_receipt_with_poi','poi_ratio','Total_earnings']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



selector= SelectKBest(f_classif, k=10)

selector.fit(features, labels)

selector.transform(features)
score = selector.scores_
features_skb = [features_list[i+1] for i in selector.get_support(indices = True )]
features_score_skb = [score[i] for i in selector.get_support(indices = True )]
feature_sorted_by_value =sorted(zip(features_skb , features_score_skb), key=lambda tup: tup[1],reverse=True)

tt = Texttable()
j=0
print "\n 10 FEATURES WITH ITS SCORE"
for i in feature_sorted_by_value:
    if j==0:
        tt.add_row(('Feature','Score'))
        j=j+1
    
    tt.add_row(i)

print tt.draw()



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.


#1
text ="GaussianNB "
pipe1 = make_pipeline(
                     MinMaxScaler(),
                     SelectKBest(),
                     GaussianNB()
                     )



params1 = dict(
               selectkbest__k=[5],
               selectkbest__score_func= [f_classif]
               )



'''

 
#2
text ="DecisionTreeClassifier"
pipe1 = make_pipeline(
                      MinMaxScaler(),
                      SelectKBest(),
                      DecisionTreeClassifier(random_state=88)
                      )


params1 = dict(
               
              selectkbest__k=[7],
              selectkbest__score_func= [f_classif],
              decisiontreeclassifier__criterion = ['gini','entropy'],
              decisiontreeclassifier__max_features =['auto']
              
             )
'''
'''
    #3
text ="SVC"
pipe1 = make_pipeline(
                     MinMaxScaler(),
                     SelectKBest(),
                     SVC(max_iter=300)
                     )



params1 = dict(selectkbest__k=[1,5,7,8,9,10],
              selectkbest__score_func= [f_classif],
              
              
              svc__C=[100000],
              svc__gamma=[0.00001],
              svc__kernel=['rbf','linear'],
              svc__class_weight=['balanced']
              
              )


'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


sss = StratifiedShuffleSplit(
                             labels_train,
                             n_iter = 20,
                             test_size = 0.25,
                             random_state = 0
                             )
'''
sss = StratifiedShuffleSplit(
                             labels_train,
                             n_iter = 1000,
                             test_size = 0.25,
                             random_state = 0
                             )
'''



clf = GridSearchCV(pipe1, param_grid=params1,cv=sss, scoring='f1')


clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)

features_k= clf.best_params_['selectkbest__k']

SKB_k=SelectKBest(f_classif, k=features_k)
SKB_k.fit_transform(features, labels)

features_scores = SKB_k.scores_

#features_pca = clf.best_estimator_
#print features_pca



prec , recall = get_precision_recall_score(labels_test,pred)

list =[]

list.append(text)
list.append(accuracy)
list.append(prec)
list.append(recall)
result_list.append(list)
from texttable import Texttable

t = Texttable()
for i in range(len(result_list)):
    
    t.add_row(result_list[i])

print t.draw()



features_selected = [features_list[i+1] for i in SKB_k.get_support(indices = True )]
features_scores_selected = [features_scores[i] for i in SKB_k.get_support(indices = True )]

print "selected features are:\n", zip(features_selected , features_scores_selected)



'''NOT INCLUDING THE NEW FEATURES IN THE FINAL TESTING'''

features_list = ['poi','salary', 'total_payments', 'exercised_stock_options','bonus',
                 'restricted_stock', 'total_stock_value','director_fees','expenses',
                 'loan_advances', 'deferred_income','restricted_stock_deferred',
                 'long_term_incentive','shared_receipt_with_poi']


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)