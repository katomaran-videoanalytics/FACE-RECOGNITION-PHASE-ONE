                                                                        FACE RECOGNITION

REFERENCE FROM : https://github.com/davidsandberg/facenet

THE SCRIPT FOR FACE RECOGNITION IS INCLUDED IN THE BELOW GIVEN REPOSITORY
REPOSITRY LINK : https://github.com/katomaran-videoanalytics/FACE-RECOGNITION-PHASE-ONE.git

TECHNOLOGIES USED:

FACE DETECTOR : MTCNN
NEURAL ARCHITECTURE : INCEPTION RESNET V1
CLASSIFIER : SVM CLASSFIER

DATASET DIRECTORY : THE DATASET CONTAINS 4 CLASSES WITH 136 IMAGES

ALIGN-DATA DIRECTORY : THE DATASET IS SEPERATED INTO TWO, ONE FOR TRAIN AND ANOTHER FOR TEST. THEREFORE THE TRAIN-DATA DIRECTORY CONTAINS 109 IMAGES FOR 4 CLASSES AND TEST-DATA DIRECTORY CONTAINS 21 IMAGES . THESE IMAGES ARE NOT ALIGNED AT FIRST,HENCE IT IS ALIGNED USING MTCNN.
	"ALIGN_DATASET_MTCNN.PY" FILE IS USED TO ALIGN BOTH THE TEST AND TRAIN DATASET.

MODEL DIRECTORY : THIS DIRECTORY IS ATTACHED WITH PRETRAINED MODELS DIRECTORY THAT CONTAINS CHECKPOINT,INDEX FILE,META FILE AND .PB FILE.

DOWNLOAD LINK:https://drive.google.com/open?id=11zWiSDA_R1fehxKNO9tPvoP0s1DZnOi8
DOWNLOAD AND SAVE THE DIRECTORY AS "modeldir" IN YOUR REPOSITRY
 
IT SHOULD BE IN TRAIN-CLASSIFIER ,TESTTING-CLASSFIER AND PREDICTION DIRECTORY. WHILE YOU RUN FURTHER SCRIPTS, PLEASE ENSURE THAT YOU HAVE THE MODEL DIR THAT WAS MENTIONED ABOVE.

TRAIN-CLASSIFIER DIRECTORY : THIS DIRECTORY IS ATTACHED WITH A SCRIPT "CLASSIFIER.PY" , WHICH IS USED TO TRAIN THE CLASSFIER. THE ALIGNED-TRAIN DATASET IS GIVEN FOR TRAINING. AFTER THE TRAINING IS COMPLETED,THIS SCRIPT WIL RETURN A .PKL FILE.

THE PIKLE FILE IS ATTACHED IN THE REPOSITRY AS KATO.PKL 
OR
YOU CAN ALSO DOWNLOAD .PKL FILE :https://drive.google.com/open?id=1V1gg6UGVLDUmvAPh1CObam2PhKOCX02X

NOTE :  1.MENTION THE .PKL FILE NAME IN ARGUMENT WHILE RUNNING THE CODE
	2.MENTION THE MODEL DIRECTORY FILE WHICH CONTAINS THE .PB FILE AND META FILE
	3.MENTION TRAIN IN THE ARGUMENT 

TESTING-CLASSFIER DIRECTORY:IT CONTAINS THE SAME SCRIPT. JUST WE ARE GOING TO CHANGE THE ARGUMENT AS "CLASSIFY" AND FEED THE ALIGNED TEST DIRECTORY AS INPUT 

IT WILL GIVE THE TEST ACCURACY RESULT

PREDICTION : IN THIS I ATTACHED A SCRIPT "PRED-ORG2.PY" TO PREDICT FROM LIVE CAMERA OR FROM ANY .MP4 FILE 
YOU CAN DOWNLOAD THE TESTING VIDEO FOR PREDICTION USING THIS LINK : https://drive.google.com/open?id=1jylvCA2-esqRWudBUpbH4mMDm8k77miY

RESULT : https://drive.google.com/open?id=1C7zUAs6kCk9oPV5-2u9QMB-P6xnjUNTJ

NOTE: YOU SHOULD HAVE THE .PKL FILE AND THE MODEL DIRECTORY WHILE RUNING THE SCRIPT.


