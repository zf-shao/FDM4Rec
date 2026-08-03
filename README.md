# FDM4Rec

## Datasets
This work utilizes the following five datasets, among which Amazon-Beauty, Amazon-Video, Amazon-Office, Amazon-Sports, and MovieLens-100K. All datasets are provided by [RecBole](https://github.com/RUCAIBox/RecSysDatasets).

All dataset files can be obtained from the cloud storage [BaiDu Cloud Drive (Password: e272)](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g#list/path=%2F).

## Run
Just run it directly：
 ```
python main.py  
 ```

Example for Beauty
 ```
config = Config(model=FDM4Rec, config_file_list=['config/config.yaml'])
 ```

## Acknowledgements
Our code implementation is based on the [RecBole](https://github.com/RUCAIBox/RecBole) and [Pytorch](https://github.com/pytorch/pytorch) frameworks.
