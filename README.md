### Random Forest algorithm for Document Classification

In this project, there are 3 python scripts as below including a dependencies python script:
<br />
In order to see how to call them in command line, use "--help" argument, as an example: 
<br />
`python rf_classification --help`

The pipeline schema for this project is as below:
<br />

<pre>


         
     ╔═════════╗ 
&#9312; &#11078;  Tika Lib            &#11078; &#9313;
     ╚═════════╝
     ╔══════════════╗ 
&#9313; &#11078;  Preprocessing       &#11078; &#9314;
     ╚══════════════╝
     ╔══════════════════╗ 
&#9314; &#11078;  Feature extraction  &#11078; &#9315;
     ╚══════════════════╝
     ╔══════════════════╗ 
&#9315; &#11078;  RF Classification   &#11078; &#9316;
     ╚══════════════════╝
     ╔═══════════╗ 
&#9316; &#11078;  Prediction          &#11078; &#9317;
     ╚═══════════╝
<br/>

<br/>
<br/>
&#9312; Original Document files (Doc, Pdf, Ppt, ...)
&#9313; Original Document Text files
&#9314; Preprocessed text documents 
&#9315; Extracted features from text documents
&#9316; Random Forest model training and classifying
&#9317; Predicting documents based on trained random forest model

</pre>



#### Dataset structure
Data (text files) can be located in different directories with multiple classes.

The number of classes and corresponding labels are defined in config1.ini file.

The class labels are defined in the "default" section, and class directories are defined in the "preprocess" and "features" section.

#### 0_rf_tika.py
In order to convert document files such as doc, docx, pdf, ppt, pptx, ... to text fies, we use "tika" library. 
Before we need to install "java" on your machine and "tika" library
There are two options to do so:

1- The "0-rf_tika.py" script, converts different original document files to text format.
it gets input, output and path to tika library in "*config1.ini*" file.


2- Another option is using tika in command line:

Then we use this command and define "input dir" as the path to the original document, and "output dir" as the path for converted text files.

`java –jar tika-1.17/tika-app/target/tika-app-1.17.jar –t –i <input_dir> -o <output_dir>`


#### 1_rf_preprocessing.py
This script, removes "sensitive words" and "stop words", and finally stemming words within each document.
The preprocessed files will be stored in the user-defined path with the same sub-folders (like the original path sub-folders).

The usage can be:

`python 1_rf_preprocessing.py ../originalTxt -o ../stemmed`

#### 2_rf_features.py
This script, contains of algorithm for extracting features.
usage for kx data can be through command line:

`python 2_rf_features.py ../kx_data`

Or by using config1.ini file.

The variable "split_plan" can take one of the following two values:
1- random_subsampling
2- cross_validation

#### 3_rf_classification.py
This script, contains of algorithm for classifying documents based on extracted features in the previous step.
usage for kx data can be with command line:

`python 3_rf_classification.py ../kx_data`

Or by using config1.ini file.

### Prediction
Prediction contains of pipeline of different scripts:

0- We need to extract the text files of the original document using tika

1- We need to preprocess the extracted text files

2- Then we need to extract the features of the documents for prediction.

3- Finally, we can predict the document based on the trained model in step 3 (classification)

In ordet to store parameters, we use "**config2.ini**" file
#### 4_0_rf_predict_tika.py
In order to convert set of document files such as doc, docx, pdf, ppt, pptx, ... to text fies, we use "tika" library. 
Before we need to install "java" on your machine and "tika" library
There are two options to do so:

1- The "0_rf_tika.py" script, converts different original document files to text format.
it gets input, output and path to tika library in "*config2.ini*" file.


2- Another option is using tika in command line:

Then we use this command and define "input dir" as the path to the original document, and "output dir" as the path for converted text files.

`java –jar tika-1.17/tika-app/target/tika-app-1.17.jar –t –i <input_dir> -o <output_dir>`


#### 4_1_rf_predict_preprocess.py
In this step, the text files will be preprocessed.

#### 4_2_rf_predict_features.py
In this script, we extract features of the list of documents to be predicted based on the extracted features of train dataset.

#### 4_3_rf_predict_prediction.py
In this script, the train model based on the 3rd step of the main pipeline (classification) and the extracted features from step 2 of prediction pipeline will be used in order to predict and label documents.




<pre>
Author: Iman Zabett
Copyright for Accenture - All right reserved
</pre>
