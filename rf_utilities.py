import os, shutil
import numpy as np

def clearFolder(folder):
    print 'Delete all files under folder %s.' % (folder)
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception, e:
            print e

def createFolder(folder):
    # create folder
    if not os.path.exists(folder):
        os.makedirs(folder)
        print 'Created folder %s ' % (folder)
    else:
        print 'Folder %s already exists. Delete all its contents.' % (folder)
        clearFolder(folder)

# Check is a directory exists
def dirExists(folder):
    try:
        if ( os.path.exists(folder) and os.path.isdir(folder)):
            return True
        else:
            return False
    except Exception, e:
        print e

# Check is a file exists
def fileExists(fpath):
    try:
        if ( os.path.exists(fpath) and os.path.isfile(fpath)):
            return True
        else:
            return False
    except Exception, e:
        print e


# Compute false positive and false negative given a confusion matrix.
# Currently this function only works for 2x2 confusion matrices
# it returns a 1 by 2 array that contains the FP and FN values
def computeFPFN(confusionMatrix):
    cm = confusionMatrix
    n, w = cm.shape[0], cm.shape[1]
    if (n !=2 or w !=2):
        print 'Error: the confusion matrix has wrong size.'
        return None

    # Compute false positives
    fp = cm[0,1]

    # Compute false negatives
    fn = cm[1,0]

    return fp, fn

def information_gain(X, y):

    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
       # print 'in _calIg(), classtCnt is'
       # print classCnt

        for c in classCnt:
            probs = classCnt[c] / float(featureTot)
        #    print 'probs=classCnt[%d]/float(featureTot) is %s\n' % (c,probs)
            entropy_x_set = entropy_x_set - probs * np.log2(probs)
         #   print 'entropy_x_set= %s\n' % entropy_x_set
            probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
            if (probs!=0):
                entropy_x_not_set = entropy_x_not_set - probs * np.log2(probs)
            # note: probs==0 means for this class c, all samples are here, no samples left in the x_not_set, so should not update entropy_x_not_set here. Will do it for c in classTotCnt and not in classCnt later 
          #  print 'entropy_x_not_set is %s\n' % entropy_x_not_set

        for c in classTotCnt:
            if c not in classCnt:
           #     print 'in _calIg in %s not in classCnt\n' % c 
                probs = classTotCnt[c] / float(tot - featureTot)
            #    print 'probs=classTotCnt[c] / float(tot - featureTot) is %s\n' % probs
             #   print 'initial entropy_x_not_set is %d' % entropy_x_not_set 
                entropy_x_not_set = entropy_x_not_set - probs * np.log2(probs)
              #  print 'entropy_x_not_set =entropy_x_not_set - p*log(p) is %s\n' % entropy_x_not_set

        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                                             +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)

    tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1

    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log2(probs)

    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain = []
    for i in range(0, len(nz[0])):
       # print '+++++++++++++++++++++++++++++++++++++i is %d' % i
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre+1, nz[0][i]):
                information_gain.append(0)
            ig = _calIg()
        #    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ig = %s' % ig
            information_gain.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1

    ig = _calIg()
    # print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ig = %s' % ig
    information_gain.append(ig)

    return np.asarray(information_gain)
