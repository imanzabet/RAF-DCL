from __future__ import division
import argparse
import os
from datetime import datetime
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import numpy as np


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
        config.read('config_2.ini')
        n_gram = config.getint('default', 'n_gram')
        num_class = config.getint('default', 'num_class')
        rootPath = config.get('features', 'in_path')
        out_path = config.get('features', 'out_path')
        vectorizer_file = config.get('features', 'vectorizer_file')

        pred_rfMatrix = config.get('features', 'pred_rfMatrix')
        pred_features = config.get('features', 'pred_features')
        trained_rfMatrix = config.get('features', 'trained_rfMatrix')
        filesPath = config.get('features', 'filesPath')



    # labels of documents
    targetFile = os.path.join(out_path, 'targetNames')
    # all files names
    fileNames = os.path.join(out_path, 'fileNames')


    # Class variables creation
    if num_class<2 or num_class>8:
        print 'Please choose a number between 2 and 8 for "Class Number"!'
        exit()

    # Feature matrix variable
    X = [None] * (n_gram+1)
    # Feature names variable
    features = [None] * (n_gram+1)

    print 'Loading all files...'

    allFiles = load_files(filesPath)
    y = allFiles.target


    print 'writing list of all files'
    outfile = open(fileNames, 'w')
    for name in allFiles.filenames:
        outfile.write((name + '\n').encode('utf-8'))
    outfile.close()

    # transforming based on trained matrix of vectorization of words for n-gram in the main pipeline
    matrix_file = os.path.join(rootPath, trained_rfMatrix)
    # X = load_svmlight_file(matrix_file)
    X = [None] * (n_gram+1)
    for s in range(1, n_gram+1):
        print 'Building %d-gram features.' % s
        # vectorizer = \
        #             CountVectorizer(ngram_range=(s,s),
        #                     analyzer="word",
        #                     tokenizer=None,
        #                     preprocessor=None,
        #                     stop_words='english',
        #                     max_features=None)

        import pickle
        filename = os.path.join(rootPath, vectorizer_file+'_'+str(s))
        # filename = 'vectorizer_1'
        fin = open(filename, 'rb')
        try:
            vectorizer = pickle.load(fin)
            fin.close()
        except EOFError:
            break
        # else:
        #     print 'READ: %s ' % (vectorizer)

        X[s] = vectorizer.transform(allFiles.data)
        features[s] = vectorizer.get_feature_names()
        y = allFiles.target

        outfile = os.path.join(out_path, pred_rfMatrix + '_' + str(s))
        dump_svmlight_file(X=X[s], y=y, f=outfile)

        outfile = open(os.path.join(out_path, pred_features + '_' + str(s)), 'w')

        for feature in features[s]:
            outfile.write((feature + '\n').encode('utf-8'))
        outfile.close()

    print 'Combine all features into a single vector.'
    from scipy.sparse import hstack
    if (n_gram==1):
        X[0] = hstack((X[1]))
    if (n_gram==2):
        X[0] = hstack((X[1], X[2]))
    if (n_gram==3):
        X[0] = hstack((X[1], X[2], X[3]))

    # dump_svmlight_file(X=X[0], y=y, f=os.path.join(out_path, pred_rfMatrix))
    data = {'X': X[0], 'Y':y}
    with open(os.path.join(out_path, pred_rfMatrix), 'wb') as fin:
        pickle.dump(data, fin, protocol=-1)

    features[0] = []

    if (n_gram==1 or n_gram == 2 or n_gram == 3):
        features[0].extend(features[1])
    if (n_gram==2 or n_gram == 3):
        features[0].extend(features[2])
    if (n_gram==3):
        features[0].extend(features[3])

    outfile = open(os.path.join(out_path, pred_features), 'w')
    for feature in features[0]:
        outfile.write((feature + '\n').encode('utf-8'))
    outfile.close()

    outfile = open(targetFile, 'w')
    for name in allFiles.target_names:
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
