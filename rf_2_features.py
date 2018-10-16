from __future__ import division
import argparse
import os
from datetime import datetime
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import numpy as np
import pandas as pd
import pickle
import rf_utilities

def main(args):
    # reading parameters:
    if (not args.config):
        print 'reading parameters from command line ...'
        n_gram = args.n_gram
        num_class = args.num_class
        # dataset directory as input of algorithm
        rootPath = args.path
        # output directory (in None use as input dir)
        out_path = args.out_path
        if (out_path==None):
            out_path = rootPath

    elif (args.config):
        print 'reading parameters from config file ...'
        from ConfigParser import SafeConfigParser

        config = SafeConfigParser()
        config.read('config_1.ini')
        n_gram = config.getint('default', 'n_gram')
        num_class = config.getint('default', 'num_class')

        # rootPath = config.get('features', 'in_path')
        out_path = config.get('features', 'out_path')
        split_plan = config.get('features', 'split_plan')
        vectorizer_file = config.get('features', 'vectorizer_file')
        rfMatrix_file = config.get('features', 'rfMatrix_file')
        rfMatrix_file_test = config.get('features', 'rfMatrix_file_test')
        featureNames = config.get('features', 'featureNames')
        targetNames = config.get('features', 'targetNames')
        trainFileNames = config.get('features', 'trainFileNames')
        testFileNames = config.get('features', 'testFileNames')
        allFileNames = config.get('features', 'allFileNames')
        indices_file = config.get('features', 'indices_file')
        num_best_features = config.getint('features', 'num_best_features')

        class_inputs = [None] * (num_class)
        in_tmp = config.get('features', 'class_inputs')
        for i, indir in enumerate(in_tmp.split()):
            class_inputs[i] = indir

        class_labels = [None] * (num_class)
        lb_tmp = config.get('default', 'class_labels')
        for i, label in enumerate(lb_tmp.split()):
            class_labels[i] = label



    # files to store the random forest feature matrix
    rfMatrix_file = os.path.join(out_path, rfMatrix_file)
    rfMatrix_file_test = os.path.join(out_path, rfMatrix_file_test)
    # labels of documents
    targetFile = os.path.join(out_path, targetNames)
    # all files names
    trainFileNames = os.path.join(out_path, trainFileNames)
    testFileNames = os.path.join(out_path, testFileNames)
    allFileNames = os.path.join(out_path, allFileNames)
    vectorizer_file = os.path.join(out_path, vectorizer_file)
    indices_file = os.path.join(out_path, indices_file)



    # Class variables creation
    if num_class<2 or num_class>8:
        print 'Please choose a number between 2 and 8 for "Class Number"!'
        exit()
    # class_labels = [None, None, None, None, None, None, None, None]
    # class_inputs = [None, None, None, None, None, None, None, None]
    # class_inputs[0] = '../orig_public/' #os.path.join(rootPath[:-9], 'orig_public/')
    # class_inputs[1] = os.path.join(rootPath[:-9], 'orig_restricted/')

    # Feature matrix variable
    X_train = [None] * (n_gram+1)
    X_testt = [None] * (n_gram + 1)
    # Feature names variable
    features = [None] * (n_gram+1)

    print 'Loading all files.'
    all_Files = [None] * (num_class)
    y = [None] * (num_class)

    for i in range(0, num_class):
        all_Files[i] = load_files(class_inputs[i])
        all_Files[i].target[:] = i
        y[i] = all_Files[i].target

    all_data = []
    all_filenames = []
    all_target = []
    all_target_names = []
    for i in range(0, num_class):
        all_data.extend(all_Files[i].data)
        all_filenames.extend(all_Files[i].filenames)
        all_target.extend(all_Files[i].target)
        all_target_names.extend(all_Files[i].target_names)

    from numpy import array
    all_Files[0].data = all_data
    all_Files[0].filenames = array(all_filenames)
    all_Files[0].target = array(all_target)
    all_Files[0].target_names = all_target_names

    all_Files = all_Files[0]
    ypp = all_Files.target


    if (split_plan == "cross_validation"): # NEED TO BE REVIESED
        test_size = 0.2
        random_state = 1
        shuffle = False
    elif (split_plan == "random_subsampling"):
        test_size = 0.2
        random_state = 1
        shuffle = True
    else:
        print 'Please input a correct split plan ...'
        exit()


    print 'writing list of all files'
    outfile = open(allFileNames, 'w')
    for name in all_Files.filenames:
        outfile.write((name + '\n').encode('utf-8'))
    outfile.close()

    from sklearn.model_selection import train_test_split
    idx = np.arange(all_Files.target.shape[0])
    x_train, \
    x_testt, \
    y_train, \
    y_testt, \
    idx_train, \
    idx_testt       = train_test_split(all_Files.data,
                               all_Files.target,
                               idx,
                               test_size=test_size,
                               random_state=random_state,
                               shuffle=True)

    # allFiles = load_files(originalDataPath)
    # y = allFiles.target

    # allFiles = all_Files
    # y = ypp
    # write indices to file
    indices = {'idx_train': idx_train, 'idx_test': idx_testt}
    with open(indices_file, 'wb') as fin:
        pickle.dump(indices, fin, protocol=-1)

    print 'writing list of all train files'
    outfile = open(trainFileNames, 'w')
    for name in all_Files.filenames[idx_train]:
        outfile.write((name + '\n').encode('utf-8'))
    outfile.close()
    print 'writing list of all test files'
    outfile = open(testFileNames, 'w')
    for name in all_Files.filenames[idx_testt]:
        outfile.write((name + '\n').encode('utf-8'))
    outfile.close()

    # creating matrix of vectorization of words for n-gram
    print 'Extract train and test features ...'
    for s in range(1, n_gram+1):
        print 'Building %d-gram features ...' % s
        vectorizer = \
                    CountVectorizer(ngram_range=(s,s),
                            analyzer="word",
                            tokenizer=None,
                            preprocessor=None,
                            stop_words='english',
                            max_features=None)

        # try:
        #     for i, doc in enumerate(x_train):
        #         XX = vectorizer.fit_transform(doc)
        #         print i
        # except:
        #     print i, doc
        X_train[s] = vectorizer.fit_transform(x_train)
        X_testt[s] = vectorizer.transform(x_testt)
        features[s] = vectorizer.get_feature_names()
        # y = allFiles.target

        with open(vectorizer_file+'_'+str(s), 'wb') as fin:
            pickle.dump(vectorizer, fin, protocol=-1)

        dump_svmlight_file(X=X_train[s], y=y_train, f=rfMatrix_file + '_' + str(s))

        outfile = open(os.path.join(out_path, featureNames + '_' + str(s)), 'w')
        for feature in features[s]:
            outfile.write((feature + '\n').encode('utf-8'))
        outfile.close()


    if (num_best_features != 0):
        from scipy.sparse import csr_matrix, coo_matrix
        for n in range(1, n_gram+1):
            if (n == 0):
                print 'Select the best %d features from combined vector.' %num_best_features
            else:
                print 'Select the best %d %d-gram features.' % (num_best_features, n)

            X_train[n] = csr_matrix(X_train[n])
            X_testt[n] = csr_matrix(X_testt[n])

            igScores = rf_utilities.information_gain(X_train[n], y_train)
            igScores = pd.Series(igScores)

            igScores.sort_values(ascending=False, inplace=True)
            bestFeatures = igScores.head(100)

            bestFeatureIndices = bestFeatures.index.tolist()
            bestFeatureNames = np.asarray(features[n])[bestFeatureIndices]
            outfile = open(featureNames + '_' + str(n), 'w')
            for feature in bestFeatureNames:
                outfile.write('%s\n' % feature)
            outfile.close()

            ### sort matrix (feature matrix X) based on bestFeatureIndices
            X_train[n] = X_train[n][:, bestFeatureIndices]
            X_testt[n] = X_testt[n][:, bestFeatureIndices]

            matrixFile = rfMatrix_file + '_' + str(n)
            dump_svmlight_file(X=X_train[n], y=y_train, f=matrixFile)

            features[n] = [features[n][j] for j in bestFeatureIndices]
            outfile = open(os.path.join(out_path, featureNames + '_' + str(n)), 'w')
            for feature in features[n]:
                outfile.write((feature + '\n').encode('utf-8'))
            outfile.close()

    print 'Combine all features into a single vector.'
    from scipy.sparse import hstack
    if (n_gram==1):
        X_train[0] = hstack((X_train[1]))
        X_testt[0] = hstack((X_testt[1]))
    if (n_gram==2):
        X_train[0] = hstack((X_train[1], X_train[2]))
        X_testt[0] = hstack((X_testt[1], X_testt[2]))
    if (n_gram==3):
        X_train[0] = hstack((X_train[1], X_train[2], X_train[3]))
        X_testt[0] = hstack((X_testt[1], X_testt[2], X_testt[3]))

    dump_svmlight_file(X=X_train[0], y=y_train, f=rfMatrix_file+'_0')
    data = {'X_testt': X_testt[0], 'y_testt': y_testt}
    with open(rfMatrix_file_test + '_0', 'wb') as fin:
        pickle.dump(data, fin, protocol=-1)

    features[0] = []
    if (n_gram == 1 or n_gram == 2 or n_gram == 3):
        features[0].extend(features[1])
    if (n_gram == 2 or n_gram==3):
        features[0].extend(features[2])
    if (n_gram==3):
        features[0].extend(features[3])

    outfile = open(os.path.join(out_path, featureNames), 'w')
    for feature in features[0]:
        outfile.write((feature + '\n').encode('utf-8'))
    outfile.close()

    outfile = open(targetFile, 'w')
    for name in all_Files.target_names:
        outfile.write('%s\n' % name)
    outfile.close()


    print 'Done.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        help='If the parameters is extracted from config file. '
                             'If True, then the command line parameters will be bypassed. '
                             'If False, then user needs to pass parameters from command line.',
                        type=bool,
                        default=True)
    parser.add_argument('-path',
                        help='The path to the project folder.')
    parser.add_argument('-n_gram',
                        help='selecting n_gram for feature selection. n_gram=3 means the algorithm'
                             'will calculate and combine 1, 2, and 3 grams together',
                        type=int,
                        default=3)
    parser.add_argument('-num_class',
                        help='defining how many classes the dataset has',
                        type=int,
                        default=2)
    parser.add_argument('-out_path',
                        help='this is the path to store feature names and featuer matrix files',
                        type=str,
                        default=None)
    args = parser.parse_args()

    startTime = datetime.now()
    main(args)
    endTime = datetime.now()

    runTime = endTime - startTime
    seconds = runTime.seconds + runTime.microseconds / 1000000
    print '\n'
    print 'The total running time is %f seconds' % seconds
