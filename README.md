# Simple handwriting recognizer (Forked and Cooked)

Original suorce code credits to: crashcourse-ai@github

**Description:**

This is a recap of what Justin and I have managed to achieve looking into machine learning to recognize handwritten characters generated from the accelerometer data.

**Prerequisites:**
```bash
$ pip3 install emnist
$ python3 -m pip install emnist scilearn opencv-python matplotlib
```
- Estimated total disk space needed after installation and decompression: `1.2Gb`
- Forked and edited from this notebook Crash course AI #5: https://colab.research.google.com/drive/1NyYH1EPpaJlMBLK0fcKYz4icaD1SNSLK#scrollTo=gfT7zKZkUyYn
- Run `main.py` within the git repo to start.
- So far has tested working on Windows Subsystem for Linux and native Linux (Arch-based and Ubuntu derivatives).

**Raw data:** 
- https://github.com/larnder/2019_06_AccelerationCamp.git `@master/doc/Alphabet/font/*.png`

**Input pre-processing:** 
- I used `Lightroom CC` to mass-crop out the letters and discard the axes. 
- Then I grayscaled the images and inverted color from positive to negative. 
- Finally, I mass-exported the images to `.jpg` with Quality = 100 (max).

**Input and source code:**
- I decided to create a different repository to house this *very experimental and alpha stage* machine learning source code to avoid further clustering the current project.
- URL: https://github.com/zasshuwu/alpharecog_neuralnetwork.git
- A copy of this note serves as the README.md file for that repository.
- The input folder is `letter_mod3/`.

**Output:**
- Output returns a string array with predictions in **lowercase** characters. Space is supported and displayed as a blank between characters.

**Training parameters:**
- Parameters and configs are noted in the `main.py` file. Feel free to meddle (locally).
