# BMVOS

This is the official PyTorch implementation of our paper:

> **Pixel-Level Bijective Matching for Video Object Segmentation**, *WACV 2022*\
> Suhwan Cho, Heansung Lee, Minjung Kim, Sungjun Jang, Sangyoun Lee\
> Link: [[WACV]](https://openaccess.thecvf.com/content/WACV2022/papers/Cho_Pixel-Level_Bijective_Matching_for_Video_Object_Segmentation_WACV_2022_paper.pdf) [[arXiv]](https://arxiv.org/pdf/2110.01644.pdf)

<img src="https://user-images.githubusercontent.com/54178929/208465815-f703ed8e-7a5d-49b3-9841-213db53549df.png" width=800>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In conventional semi-supervised VOS methods, the query frame pixels select the best-matching pixels in the reference frame and transfer the information from those pixels without
any consideration of reference frame options. As there is no limitation to the number of reference frame pixels being referenced, background distractors in the query frame will
get high foreground scores and can disrupt the prediction. To mitigate this issue, we introduce a **bijective matching mechanism** to find the best matches from the query frame to
the reference frame and also vice versa. In addition, to take advantage of the property of a video that an object usually occupies similar positions in consecutive frames, we 
propose a **mask embedding module**.


## Preparation
1\. Download [DAVIS](https://davischallenge.org/davis2017/code.html) and [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data) for network training and testing.

2\. Download [Custom Split](https://drive.google.com/drive/folders/1R5Z0aQQw2lvsoAlqtHLjY4RYNXg7SJkX?usp=drive_link) for YouTube-VOS training samples for network validation.

3\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
Please follow the instructions in [TBD](https://github.com/suhwan-cho/TBD).


## Testing
1\. Move to "run.py" file.

2\. Select a pre-trained model.

3\. Run **BMVOS** testing!!
```
python run.py
```


## Attachments
[pre-trained model (davis)](https://drive.google.com/file/d/1msMoNOYeIK7nUSpuI7djy-EN7GJtIcfr/view?usp=sharing)\
[pre-trained model (ytvos)](https://drive.google.com/file/d/1lMpXh2MSLbcgdCwfusAchzfuRdEvXHqU/view?usp=sharing)\
[pre-computed results](https://drive.google.com/file/d/1AokjFfA8Gb65dIQ3T3V4lM5CuOfKuIqm/view?usp=drive_link)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: chosuhwan@yonsei.ac.kr
```
