from __future__ import division
import argparse
import rf_utilities
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from time import localtime, strftime
from datetime import datetime
from sklearn.externals import joblib
from time import time

class ExpRecord(object):
    startTime = ''
    endTime = ''
    positiveClass = ''
    weight = 1.0
    n_estimators = 120
    criterion = ''
    rfNegPrecision = 0.0
    rfNegRecall = 0.0
    rfNegFScore = 0.0
    rfNegSupport = 0
    rfPosPrecision = 0.0
    rfPosRecall = 0.0
    rfPosFScore = 0.0
    rfPosSupport = 0
    rfFP = 0
    rfFN = 0
    rfFScore = 0.1

    precision = 0
    recall = 0
    fbeta_score = 0
    support = 0
    accu = 0



    def printTitles(self):
        return 'startDate startTime endDate endTime positiveClass n_estimators criterion ' + \
                'rfNegPrecision rfNegRecall rfNegFScore rfNegSupport '+\
                'rfPosPrecision rfPosRecall rfPosFScore rfPosSupport ' \
                'rfFP rfFN rfFScore \n'

    def output(self):
        # return '%s %s %s ' \
        #        '%d, %s ' \
        #        '%.3f %.3f %.3f %d ' \
        #        '%.3f %.3f %.3f %d ' \
        #        '%d %d %.3f \n' \
        #        % (  self.startTime, self.endTime, self.positiveClass,
        #             self.n_estimators, self.criterion,
        #             self.rfNegPrecision, self.rfNegRecall, self.rfNegFScore, self.rfNegSupport,
        #             self.rfPosPrecision, self.rfPosRecall, self.rfPosFScore, self.rfPosSupport,
        #             self.rfFP, self.rfFN, self.rfFScore)
        out1 =  'startTime: %s \tendTime: %s \tpositiveClass: %s\n' %(str(self.startTime) , str(self.endTime) , str(self.positiveClass))
        out2 =  'n_estimators: %s \tcriterion: %s\n' %(str(self.n_estimators) , self.criterion)
        out3 =  'rfNegPrecision: %s \trfNegRecall: %s \trfNegFScore: %s \trfNegFScore: %s\n' %(str(self.rfNegPrecision) , str(self.rfNegRecall) , str(self.rfNegFScore) , str(self.rfNegSupport))
        out4 =  'rfPosPrecision: %s \trfPosRecall: %s \trfPosFScore: %s \trfPosSupport: %s\n' %(str(self.rfPosPrecision) , str(self.rfPosRecall) , str(self.rfPosFScore) , str(self.rfPosSupport))
        out5 =  'rfFP:%s \trfFN: %s \trfFscore: %s\n' %(str(self.rfFP), str(self.rfFN), str(self.rfFScore))
        out6 =  'Accuracy: %s \tPrecision: %s \tRecall: %s \tF_Score: %s \t Support: %s\n' %(str(self.accu), str(self.precision), str(self.recall), str(self.fbeta_score), str(self.support))
        return '\n'+'\n'+out1+out2+out3+out4+out5+out6


def computeFScore(y_true, y_pred):
    f = metrics.fbeta_score(y_true, y_pred, average='binary', beta=0.5)
    return f

def classify(clf, X, y, idx,
             targetNames, docNames,
             allPredictions, tags,
             wrongPredictions,
             classifier,
             fp_filename,
             fn_filename):
    print '\n'
    print 'Perform %s classification' % classifier

    predicted = clf.predict(X)
    cm = metrics.confusion_matrix(y, predicted)
    fp, fn = rf_utilities.computeFPFN(cm)
    print 'False positives = %d, False negatives = %d' % (fp, fn)

    results = metrics.precision_recall_fscore_support(y, predicted)
    report = metrics.classification_report(y, predicted, target_names=targetNames)
    lines = report.split('\n')
    line = lines[-2]
    row_data = line.split('     ')
    accu = float(row_data[1])
    print 'Prediction Accuracy is %.2f' % accu

    fscore = computeFScore(y, predicted)

    doc_idx = pd.Series(idx, name='Doc_Index')
    true_class = pd.Series(y, name='True_Class')
    pred_class = pd.Series(predicted, name='Pred_Class')
    predictions_df = pd.concat([doc_idx, true_class, pred_class], axis=1)
    predictions_df.to_csv(allPredictions + '_' + classifier)

    # Printing FP and FN documents
    fnf = open(fn_filename, 'w')
    fpf = open(fp_filename, 'w')
    for item in predictions_df.values:
        if item[1] > item[2]:
            # print 'FN Doc:', docNames[int(item[0])]
            fnf.write(docNames[int(item[0])] + '\n')

        if item[1] < item[2]:
            # print 'FP Doc:', docNames[int(item[0])]
            fpf.write(docNames[int(item[0])] + '\n')
    fnf.close()
    fpf.close()

    tags = tags + '_' + classifier
    outfile = open(tags, 'w')
    for i in range(0, len(y)):
        doc_idx = idx[i]
        if predicted[i] == 0:
            label = 'Public'
        if predicted[i] == 1:
            label = 'Restricted'
        outfile.write('%s#label:%s\n' % (docNames[doc_idx].rstrip(), label))
    outfile.close()

    wrongPredictions = wrongPredictions + '_' + classifier
    outfile = open(wrongPredictions, 'w')
    outfile.write('%s\t%s\t%s\n' % ('doc_name', 'true_class', 'prediction'))

    for i in range(0, len(y)):
        if (y[i] != predicted[i]):
            doc_idx = idx[i]
            outfile.write('%s\t%d\t%d\n' % (docNames[doc_idx].rstrip(), y[i], predicted[i]))
    outfile.close()
    return results, fp, fn, fscore, accu, metrics.precision_recall_fscore_support(y, predicted)


def main(args):
    ### Parameters
    if (not args.config):
        print 'reading parameters from command line ...'
        n_gram = args.n_gram
        num_features = args.num_features
        # RF algo parameters
        n_estimators = args.n_estimators
        print 'n_estimators = %d' %n_estimators
        criterion = args.criterion
        print 'criterion = %s' %criterion
        # class_weight = args.class_weight
        # import json
        # def ascii_encode_dict(data):
        #     ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
        #     return dict(map(ascii_encode, pair) for pair in data.items())
        #
        # class_weight = json.loads(args.class_weight, object_hook=ascii_encode_dict)
        class_weight = args.class_weight

        # Plan of splitting dataset
        split_plan = args.split_plan
        rootPath = args.path


    elif (args.config):
        print 'reading parameters from config file ...'
        from ConfigParser import SafeConfigParser

        config = SafeConfigParser()
        config.read('config_1.ini')
        n_gram = config.getint('default', 'n_gram')
        num_class = config.getint('default', 'num_class')

        ### From feature step
        targetNames = config.get('features', 'targetNames')
        rfMatrix_file = config.get('features', 'rfMatrix_file')
        rfMatrix_file_test = config.get('features', 'rfMatrix_file_test')
        trainFileNames = config.get('features', 'trainFileNames')
        testFileNames = config.get('features', 'testFileNames')
        allFileNames = config.get('features', 'allFileNames')
        indices_file = config.get('features', 'indices_file')

        ### From classification step
        num_features = config.getint('classification', 'num_features')
        rootPath = config.get('classification', 'in_path')
        out_path = config.get('classification', 'out_path')
        n_estimators = config.getint('classification', 'n_estimators')
        criterion = config.get('classification', 'criterion')
        class_weight = eval(config.get('classification', 'class_weight'))
        fp_filename = config.get('classification', 'fp_filename')
        fn_filename = config.get('classification', 'fn_filename')
        top_features = config.get('classification', 'top_features')
        targetNames2 = config.get('classification', 'targetNames2')
        expRecords = config.get('classification', 'expRecords')
        tags = config.get('classification', 'tags')
        wrongPredictions = config.get('classification', 'wrongPredictions')
        predictions = config.get('classification', 'predictions')
        trainFiles = config.get('classification', 'trainFiles')
        testFiles = config.get('classification', 'testFiles')
	overallResult = config.get('classification', 'overallResult')



    if not (num_features == 100 or
                    num_features == 200 or
                    num_features == 500):
        print 'The number of features must be 100, 200 or 500'
        exit()



    # input directory for dataset and matrix of features from previous step
    targetFile = os.path.join(rootPath, targetNames)
    targetFile2 = os.path.join(rootPath, targetNames2)
    # output file to write the algorithm results
    recordFile = os.path.join(rootPath, expRecords)
    # feature Matrix created in previous step (rf_features)
    rfMatrix_file = os.path.join(rootPath, rfMatrix_file)
    rfMatrix_file_test = os.path.join(rootPath, rfMatrix_file_test)
    # list of documents
    allFileNames = os.path.join(out_path, allFileNames)
    # csv files to
    allPredictions = os.path.join(rootPath, predictions)
    tags = os.path.join(rootPath, tags)
    wrongPredictions = os.path.join(rootPath, wrongPredictions)
    trainFiles = os.path.join(rootPath, trainFiles)
    testFiles = os.path.join(rootPath, testFiles)
    fp_filename = os.path.join(rootPath, fp_filename)
    fn_filename = os.path.join(rootPath, fn_filename)
    top_features = os.path.join(rootPath, top_features)
    indices_file = os.path.join(out_path, indices_file)
    overallResult = os.path.join(out_path, overallResult)


    X_train = [None] * (n_gram+1)

    target = None
    for s in range(0, 4):
        ### feature_matrix
        infile = rfMatrix_file + '_'+str(s)
        data = load_svmlight_file(infile)
        X_train[s] = data[0]
        if s==0:
            y_train = data[1]
    import pickle
    with open(rfMatrix_file_test + '_0', 'rb') as fin:
        data = pickle.load(fin)
        fin.close()
    X_testt = data['X_testt'].tocsr()
    y_testt = data['y_testt']



    import pickle
    with open(indices_file, 'rb') as fin:
        indices = pickle.load(fin)
        fin.close()
    idx_train = indices['idx_train']
    idx_testt = indices['idx_test']

    # automatically extract n_gram from extracted feature
    # if (not 'n_gram' in locals()):
    #     if X[3] != None:
    #         n_gram = 3
    #     elif X[2] != None:
    #         n_gram = 2
    #     elif X[1] != None:
    #         n_gram = 1
    #     else:
    #         print 'n_gram cannot be defined.'
    #         exit()
    #     print 'n_gram is: %d' %n_gram
    # y = np.copy(target)


    ### training model
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,
                                class_weight=class_weight,
                                 n_jobs=-1)


    # idx = np.arange(X[0].shape[0])
    # X_train, X_test, y_train, y_test, idx_train, idx_test = \
    #                             train_test_split(X[0], y, idx,
    #                                              test_size=test_size,
    #                                              random_state=random_state,
    #                                              shuffle=shuffle)
    #
    #
    #
    #
    #
    #

    filename = os.path.join(rootPath, 'RF_Model.pkl')
    t0 = time()
    print('training random forest ...')
    clf = clf.fit(X_train[0], y_train)
    tf = time() - t0
    print ('total training time: ' + str(tf))
    joblib.dump(clf, filename)


    ### classification of test data
    record = ExpRecord()
    # reading target file created in previous step, contains set of class labels
    infile = open(targetFile, 'r')
    targetNames = [name.rstrip('\n') for name in infile]
    infile.close()

    infile = open(allFileNames, 'r')
    docNames = [name for name in infile]
    infile.close()

    # outfile = open(trainFiles, 'w')
    # for i in idx_train:
    #     outfile.write('%s\n' % (docNames[i].rstrip()))
    # outfile.close()

    # outfile = open(testFiles, 'w')
    # for i in idx_test:
    #     outfile.write('%s\n' % (docNames[i].rstrip()))
    # outfile.close()

    if not (args.positive in targetNames):
        print 'Warning: did not find the specified positive class! Keep the target as it is.'
    else:
        print ('positive class is:', args.positive)
        if (targetNames[1] == args.positive):
            print ('')
        elif (targetNames[0] == args.positive):
            y[target == 0] = 1
            y[target == 1] = 0
            t = targetNames[0]
            targetNames[0] = targetNames[1]
            targetNames[1] = t

    outfile = open(targetFile2, 'w')
    for i in range(0, len(targetNames)):
        if targetNames[i] == 'public':
            printName = 'Unrestricted'
        if targetNames[i] == 'restricted':
            printName = 'Accenture Internal Use Only'
        print 'Class %d: %s' % (i, printName)
        outfile.write(targetNames[i] + '\n')
    outfile.close()



    if rf_utilities.fileExists(recordFile):
        f = open(recordFile, 'a')
    else:
        f = open(recordFile, 'w')
    # else:
    #     f = open(recordFile, 'w')
    #     f.write(record.printTitles())

    rfResults, rfFP, rfFN, rfFScore, rfAccuarcy, rf_prfsMatrix = \
                            classify(clf, X_testt, y_testt, idx_testt,
                                        targetNames=targetNames,
                                        docNames=docNames,
                                        allPredictions=allPredictions,
                                        tags=tags,
                                        wrongPredictions=wrongPredictions,
                                        classifier='random_forest',
                                        fp_filename=fp_filename,
                                        fn_filename=fn_filename)

    record.startTime = strftime("%Y-%m-%d %H:%M:%S", localtime())
    record.endTime = strftime("%Y-%m-%d %H:%M:%S", localtime())

    record.positiveClass = args.positive
    record.criterion = args.criterion
    record.n_estimator = args.n_estimators
    record.weight = args.class_weight
    # record.rfNegPrecision = rfResults[0][0]
    # record.rfNegRecall = rfResults[1][0]
    record.rfNegFScore = rfResults[2][0]
    record.rfNegSupport = rfResults[3][0]
    # record.rfPosPrecision = rfResults[0][1]
    # record.rfPosRecall = rfResults[1][1]
    record.rfPosFScore = rfResults[2][1]
    record.rfPosSupport = rfResults[3][1]
    record.rfFP = rfFP
    record.rfFN = rfFN
    record.rfFScore = rfFScore
    record.precision = rf_prfsMatrix[0]
    record.recall = rf_prfsMatrix[1]
    record.fbeta_score = rf_prfsMatrix[2]
    record.support = rf_prfsMatrix[3]
    record.accu = rfAccuarcy

    f.write(record.output())
    f.close()

    ### Stats
    target = list(y_train)
    b = list(y_testt)
    target.extend(b)
    print ('Total number of Documents: %d' %target.__len__())
    for i, name in enumerate(targetNames):
        print ("total number of class %s is: %d" % (name, target.count(i)))

    s = ['','','','']
    s[0] = 'Total number of RF False Positive:%d' %rfFP
    s[1] = 'Total number of RF False Negative:%d' %rfFN
    s[2] = 'Rate of False Positive: %f' %(rfFP/target.count(0))
    s[3] = 'Rate of False Negative: %f' %(rfFN/target.count(1))

    filename = open(overallResult, 'w')
    for line in s:
        filename.write(line+'\n')
    filename.close()

    ### finding best features
    def rf_feature_names(clf):
        feature_names = [None] * (n_gram+1)
        for s in range(1, n_gram+1):
            file = open(os.path.join(rootPath, 'featureNames_' + str(s)), 'r')
            feature_names[s] = list()
            for line in file:
                feature_names[s].append(line.rstrip('\n'))
            file.close()

        if(n_gram==1):
            feature_names[0] = feature_names[1]
        if(n_gram==2):
            feature_names[0] = feature_names[1] + \
                           feature_names[2]
        if(n_gram==3):
            feature_names[0] = feature_names[1] + \
                           feature_names[2] + \
                           feature_names[3]

        srt = clf.feature_importances_.argsort()[::-1]
        best_Feature_Names = [feature_names[0][j] for j in srt]
        return best_Feature_Names
    bestFeatureNames = rf_feature_names(clf=clf)
    # print '\n best %d top features:' %num_features
    # print bestFeatureNames[:num_features]
    filename = open(top_features, 'w')
    for line in bestFeatureNames[:num_features]:
        filename.write(line+'\n')
    filename.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        help='If the parameters is extracted from config file. '
                             'If True, then the command line parameters will be bypassed. '
                             'If False, then user needs to pass parameters from command line.',
                        type=bool,
                        default=True)
    parser.add_argument('-path', help='The path to the project folder.')
    parser.add_argument('-split_plan',
                        help='The starategy of splitting data for train/test and evaluation.'
                             ' It can be "cross_validation" or "random_subsampling"',
                        type=str,
                        default="cross_validation")
    parser.add_argument('-n_estimators',
                        help='The number of estimators using in random forest algo; default=120',
                        type=int, default=120)
    parser.add_argument('-criterion',
                        help='The criterion for splitting features either "gini" or "entropy"; default="gini"',
                        type=str, default="gini")
    parser.add_argument('-class_weight',
                        help='To define whether the class samples are equal in number.'
                             '"balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data,'
                             '"balanced_subsample" mode is the same as "balanced" except that weights are computed based on the bootstrap sample for every tree grown.',
                        type=str, default=None)
    parser.add_argument('-positive', help='Name of positive class. Default=restricted',
                        type=str, default='restricted')
    parser.add_argument('-num_features', help='Number of best features to be written in a file. '
                                              'The number must be 100, 200 or 500',
                        type=int, default=100)
    # cannot be done automatically since we need to read feauture files first
    # parser.add_argument('-n_gram',
    #                     help='selecting n_gram for feature selection. n_gram=3 means the algorithm'
    #                          'will calculate and combine 1, 2, and 3 grams together',
    #                     type=int, default=3)
    args = parser.parse_args()

    startTime = datetime.now()
    main(args)
    endTime = datetime.now()

    runTime = endTime - startTime
    seconds = runTime.seconds + runTime.microseconds / 1000000
    print '\n'
    print 'The total running time is %f seconds' % seconds
