# BMVOS

This is the official PyTorch implementation of our paper:

> **Pixel-Level Bijective Matching for Video Object Segmentation**, *WACV 2022*\
> Suhwan Cho, Heansung Lee, Minjung Kim, Sungjun Jang, Sangyoun Lee\
> Link: [[WACV]](https://openaccess.thecvf.com/content/WACV2022/papers/Cho_Pixel-Level_Bijective_Matching_for_Video_Object_Segmentation_WACV_2022_paper.pdf) [[arXiv]](https://arxiv.org/abs/2110.01644)

<img src="https://github.com/user-attachments/assets/6ff1d000-c251-4cbf-a556-4e60a848c256" width=800>

You can also explore other related works at [awesome-video-object segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
In conventional semi-supervised VOS methods, the query frame pixels select the best-matching pixels in the reference frame and transfer the information from those pixels without
any consideration of reference frame options. As there is no limitation to the number of reference frame pixels being referenced, background distractors in the query frame will
get high foreground scores and can disrupt the prediction. To mitigate this issue, we introduce a **bijective matching mechanism** to find the best matches from the query frame to
the reference frame and also vice versa. In addition, to take advantage of the property of a video that an object usually occupies similar positions in consecutive frames, we 
propose a **mask embedding module**.


## Setup
1\. Download the datasets: [DAVIS](https://davischallenge.org/davis2017/code.html),
[YouTube-VOS](https://competitions.codalab.org/competitions/19544#participate-get-data).

2\. Download our [custom split](https://drive.google.com/drive/folders/14FcZXKjqIVoO375w3_bH6YcQI9GuKYIf?usp=drive_link) for the YouTube-VOS training set.


## Running

### Training
Please follow the instructions in [TBD](https://github.com/suhwan-cho/TBD).


### Testing
Run BMVOS with:
```
python run.py
```


Verify the following before running:\
✅ Testing dataset selection\
✅ GPU availability and configuration\
✅ Pre-trained model path


## Attachments
[Pre-trained model (davis)](https://drive.google.com/file/d/16FmdVvTn9BKAr2X1QbJhcMOvQ4av_JGw/view?usp=drive_link)\
[Pre-trained model (ytvos)](https://drive.google.com/file/d/1Gbv9SfNkJf08kd2ZWiRdEjWTP0em20f2/view?usp=drive_link)\
[Pre-computed results](https://drive.google.com/file/d/18Pf6n7uh0Ima5tcu-rmbGJ2GnyPEuixK/view?usp=drive_link)


## Contact
Code and models are only available for non-commercial research purposes.\
For questions or inquiries, feel free to contact:
```
E-mail: suhwanx@gmail.com
```
