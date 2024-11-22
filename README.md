# BMVOS

This is the official PyTorch implementation of our paper:

> **Pixel-Level Bijective Matching for Video Object Segmentation**, *WACV 2022*\
> Suhwan Cho, Heansung Lee, Minjung Kim, Sungjun Jang, Sangyoun Lee\
> Link: [[WACV]](https://openaccess.thecvf.com/content/WACV2022/papers/Cho_Pixel-Level_Bijective_Matching_for_Video_Object_Segmentation_WACV_2022_paper.pdf) [[arXiv]](https://arxiv.org/pdf/2110.01644.pdf)

<img src="https://github.com/user-attachments/assets/812c4399-5afb-4b12-9dbf-5628fdbe02f3" width=750>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In conventional semi-supervised VOS methods, the query frame pixels select the best-matching pixels in the reference frame and transfer the information from those pixels without
any consideration of reference frame options. As there is no limitation to the number of reference frame pixels being referenced, background distractors in the query frame will
get high foreground scores and can disrupt the prediction. To mitigate this issue, we introduce a **bijective matching mechanism** to find the best matches from the query frame to
the reference frame and also vice versa. In addition, to take advantage of the property of a video that an object usually occupies similar positions in consecutive frames, we 
propose a **mask embedding module**.


## Preparation
1\. Download 
[DAVIS](https://davischallenge.org/davis2017/code.html)
and [YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data)
from the official websites.

2\. Download our [custom split](https://drive.google.com/drive/folders/14FcZXKjqIVoO375w3_bH6YcQI9GuKYIf?usp=drive_link) for the YouTube-VOS training set.

3\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
Please follow the instructions in [TBD](https://github.com/suhwan-cho/TBD).


## Testing
1\. Open the "run.py" file.

2\. Choose a pre-trained model.

3\. Start **BMVOS** testing!
```
python run.py
```


## Attachments
[pre-trained model (davis)](https://drive.google.com/file/d/16FmdVvTn9BKAr2X1QbJhcMOvQ4av_JGw/view?usp=drive_link)\
[pre-trained model (ytvos)](https://drive.google.com/file/d/1Gbv9SfNkJf08kd2ZWiRdEjWTP0em20f2/view?usp=drive_link)\
[pre-computed results](https://drive.google.com/file/d/18Pf6n7uh0Ima5tcu-rmbGJ2GnyPEuixK/view?usp=drive_link)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: suhwanx@gmail.com
```
