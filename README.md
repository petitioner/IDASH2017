# IDASH2017

IDASH2017 is a project for implementing our Logistic Regression Traning on  encrypted datasets (Privacy-Preserving Logistic Regression Training with A Faster Gradient Variant )


## How to run this program? 

### Dependencies

On a Ubuntu cloud, our implementation requires the following libraries in order:
* `g++`:      
```sh
               # apt install g++ 
```

* `make`:       
```sh
                # apt install make
```

* `m4`: #        
```sh
                 # apt install m4
```

* `GMP`(ver. 6.1.2):      
```sh
                           # cd gmp-x.x.x  
                           # ./configure --enable-cxx  
                           # make
                           # make install
                           # ldconfig
```

* `NTL`(ver. 11.3.0): 
```sh
                     # cd ntl-x.x.x
                     # cd src
                     # ./configure NTL_THREADS=on NTL_THREAD_BOOST=on NTL_EXCEPTIONS=on
                     # make
                     # make install
```

### Running IDASH2017

You need to configure and build the CNNinference project. 

After that, in the 'Debug' folder, you can run our project by the following command lines:

```sh
# make clean
# make all
# ./MyIDASH2017
``` 

You can change the source codes and then repeat the above lines to debug your own project.

## Running a test source code

In the 'Debug' folder, you can find the C++ running results for six datasets:   

        'MyIDASH2017ArchiveFile20220429_SetNumThreads(36)_kdeg5_numIter3_fold10_idash18x1579.txt_B_nohup.out'  
        
        'MyIDASH2017ArchiveFile20220429_SetNumThreads(36)_kdeg5_numIter3_fold5_edin.txt_F_nohup.out'  
        'MyIDASH2017ArchiveFile20220429_SetNumThreads(36)_kdeg5_numIter3_fold5_lbw.txt_B_nohup.out'  
        'MyIDASH2017ArchiveFile20220429_SetNumThreads(36)_kdeg5_numIter3_fold5_nhanes3.txt_D_nohup.out'  
        'MyIDASH2017ArchiveFile20220429_SetNumThreads(36)_kdeg5_numIter3_fold5_pcs.txt_D_nohup.out'  
        'MyIDASH2017ArchiveFile20220429_SetNumThreads(36)_kdeg5_numIter3_fold5_uis.txt_E_nohup.out'  
  
        
        



            
            
    

