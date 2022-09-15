# Plant-seedings-Classification

**Plant-seedings-Classification task on kaggle with pytorch**
The highest score on kaggle is `0.93198`
The details of model and training,check `train_Residual_Network.py`
* * *

## Environment
The environment of the project is constructed by `conda`
To check the environment setup, see `requirements.txt`
Training on GTX 1080Ti with CUDA version **11.4**
* * *

##Training
The train set contains **3800** images and valid set contains **950** images
Executing `train_Residual_Network.py`,we can get the best weight model named `Seedings_Classification_Residual_Network_best.ckpt`
and its logs file `Seedings_Classification_Residual_Network_log`  

The change of training/validation loss are shown as the cureve graph below 
![](https://github.com/weic0813/plant-seedings-task/blob/main/figures/Loss_Residual_Network.png?raw=true)
![](https://github.com/weic0813/plant-seedings-task/blob/main/figures/Accs_Residual_Network.png?raw=true)
* * *

##Testing
After getting the trained model,we can executing `test_acc_RN.py` to make a csv file for submission
like my `submission_Residual_Network.csv`.
Then we can make a submission on Kaggle.
* * *

##Reference URL
[https://ithelp.ithome.com.tw/articles/10222575](https://ithelp.ithome.com.tw/articles/10222575)

https://colab.research.google.com/drive/15hMu9YiYjE_6HY99UXon2vKGk2KwugWu