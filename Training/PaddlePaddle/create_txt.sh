###
 # @Author: TJUZQC
 # @Date: 2020-12-09 12:26:25
 # @LastEditors: TJUZQC
 # @LastEditTime: 2020-12-09 12:26:26
 # @Description: None
### 
#!/bin/bash

dataset_path="$1"
if [ ! -n $dataset_path ]; then
    cd $dataset_path
fi

if [ -f img_train.zip ] && [ -f lab_train.zip ]; then
    if [ -f train_list.txt ]; then
        echo "File train_list.txt has existed."
    elif [ -f val_list.txt ]; then
        echo "File val_list.txt has existed."
    else 
        echo "unzip img_train.zip..."
        unzip img_train.zip > /dev/null 2>&1
        echo "unzip lab_train.zip..."
        unzip lab_train.zip > /dev/null 2>&1
        
        find img_train -type f | sort > train.ccf.tmp
        find lab_train -type f | sort > train.lab.ccf.tmp
        paste -d " " train.ccf.tmp train.lab.ccf.tmp > all.ccf.tmp
        
        awk '{if (NR % 50 != 0) print $0}' all.ccf.tmp > train_list.txt
        awk '{if (NR % 50 == 0) print $0}' all.ccf.tmp > val_list.txt
    
        rm *.ccf.tmp
        echo "Create train_list.txt and val_list.txt."
    fi
fi

if [ -f img_testA.zip ]; then
    if [ -f testA_list.txt ]; then
        echo "File testA_list.txt has existed."
    else
        echo "unzip img_testA.zip..."
        unzip img_testA.zip > /dev/null 2>&1
        find img_testA -type f | sort > testA_list.txt
        echo "Create testA_list.txt."
    fi
fi

if [ -f img_testB.zip ]; then
    if [ -f testB_list.txt ]; then
        echo "File testB_list.txt has existed."
    else
        echo "unzip img_testB.zip..."
        unzip img_testB.zip > /dev/null 2>&1
        find img_testB -type f | sort > testB_list.txt
        echo "Create testB_list.txt."
    fi
fi
