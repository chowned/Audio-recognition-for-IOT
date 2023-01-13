#!/bin/bash
# me lazy

echo "These are the possible steps available, choose among them."
echo "Be careful that no error management is done"
echo "If run for the first time you have to run all steps sequentially with '1 2 3 4 5'"
echo ""
echo "Step 1 : Creating Directory Tree"
echo "Step 2 : Making sure the directories are totally empty"
echo "Step 3 : Running the dataset preparation steps"
echo "Step 4 : Running the model fitting on the prepared dataset"
echo ""
read steps
# clear

testDatasetName="Test_Dataset_Truncated"
train_Dataset_Truncated="Train_Dataset_Truncated"

# echo "Step 1 : Creating Directory Tree"
if [[ $steps == *1* ]];then
    echo "Step 1 : Creating Directory Tree"
    mkdir -p ./{${testDatasetName}/,${train_Dataset_Truncated}/}
fi

# echo "Step 2 : Making sure the directories are totally empty"
if [[ $steps == *2* ]];then
    echo "Step 2 : Making sure the directories are totally empty"
    rm ./$testDatasetName/*  &> /dev/null
    rm ./$train_Dataset_Truncated/* &> /dev/null
fi

# echo "Step 3 : Running the dataset preparation steps"
if [[ $steps == *3* ]];then
    echo "Step 3 : Running the dataset preparation steps"
    python3 ./1_Preparing_Dataset_Train.py
    python3 ./2_Preparing_Dataset_Evaluation.py
    python3 ./3_Preparing_Label_Set.py
fi

# python3 ./4_Model_Training.py
if [[ $steps == *4* ]];then
    echo "Step 4 : Running the model fitting on the prepared dataset"
    python3 ./4_Model_Training.py
fi